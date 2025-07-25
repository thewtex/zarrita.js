import type { Readable } from "@zarrita/storage";
import { create_codec_pipeline } from "../codecs.js";
import type { Location } from "../hierarchy.js";
import type { Chunk, DataType } from "../metadata.js";
import { assert, type ShardingCodecMetadata, get_strides } from "../util.js";

const MAX_BIG_UINT = 18446744073709551615n;

export function create_sharded_chunk_getter<Store extends Readable>(
  location: Location<Store>,
  shard_shape: number[],
  encode_shard_key: (coord: number[]) => string,
  sharding_config: ShardingCodecMetadata["configuration"]
) {
  assert(location.store.getRange, "Store does not support range requests");
  let get_range = location.store.getRange.bind(location.store);
  let index_shape = shard_shape.map(
    (d, i) => d / sharding_config.chunk_shape[i]
  );
  let index_codec = create_codec_pipeline({
    data_type: "uint64",
    shape: [...index_shape, 2],
    codecs: sharding_config.index_codecs,
  });

  let cache: Record<string, Chunk<"uint64"> | null> = {};
  return async (chunk_coord: number[]) => {
    let shard_coord = chunk_coord.map((d, i) => Math.floor(d / index_shape[i]));
    let shard_path = location.resolve(encode_shard_key(shard_coord)).path;

    let index: Chunk<"uint64"> | null;
    if (shard_path in cache) {
      index = cache[shard_path];
    } else {
      let checksum_size = 4;
      let index_size = 16 * index_shape.reduce((a, b) => a * b, 1);
      let bytes = await get_range(shard_path, {
        suffixLength: index_size + checksum_size,
      });
      index = cache[shard_path] = bytes
        ? await index_codec.decode(bytes)
        : null;
    }

    if (index === null) {
      return undefined;
    }

    let { data, shape, stride } = index;
    let linear_offset = chunk_coord
      .map((d, i) => d % shape[i])
      .reduce((acc, sel, idx) => acc + sel * stride[idx], 0);

    let offset = data[linear_offset];
    let length = data[linear_offset + 1];
    // write null chunk when 2^64-1 indicates fill value
    if (offset === MAX_BIG_UINT && length === MAX_BIG_UINT) {
      return undefined;
    }
    return get_range(shard_path, {
      offset: Number(offset),
      length: Number(length),
    });
  };
}

/**
 * A builder for creating shards from multiple chunks.
 * This handles the logic of collecting chunks and building the final shard.
 */
export class ShardBuilder<D extends DataType> {
  #chunks: Map<string, Chunk<D>> = new Map();
  #shard_shape: number[];
  #inner_chunk_shape: number[];
  #index_shape: number[];
  #inner_codec: ReturnType<typeof create_codec_pipeline<D>>;
  #index_codec: ReturnType<typeof create_codec_pipeline<"uint64">>;

  constructor(
    shard_shape: number[],
    sharding_config: ShardingCodecMetadata["configuration"],
    data_type: D,
    _fill_value: any
  ) {
    this.#shard_shape = shard_shape;
    this.#inner_chunk_shape = sharding_config.chunk_shape;

    // Calculate index shape - number of chunks in each dimension of the shard
    this.#index_shape = this.#shard_shape.map((shard_dim, i) =>
      Math.ceil(shard_dim / this.#inner_chunk_shape[i])
    );

    // Create codec pipeline for individual chunks within the shard
    this.#inner_codec = create_codec_pipeline({
      data_type,
      shape: this.#inner_chunk_shape,
      codecs: sharding_config.codecs,
    });

    // Create codec pipeline for shard index
    this.#index_codec = create_codec_pipeline({
      data_type: "uint64",
      shape: [...this.#index_shape, 2],
      codecs: sharding_config.index_codecs,
    });
  }

  /**
   * Add a chunk to this shard at the given inner chunk coordinates.
   */
  setChunk(inner_chunk_coord: number[], chunk: Chunk<D>) {
    const key = inner_chunk_coord.join(",");
    this.#chunks.set(key, chunk);
  }

  /**
   * Build the complete shard bytes.
   */
  async build(): Promise<Uint8Array> {
    const chunks_data: Uint8Array[] = [];
    const index_entries: bigint[] = [];

    // Calculate total number of inner chunks in the shard
    const total_chunks = this.#index_shape.reduce((a, b) => a * b, 1);

    // Iterate through all possible inner chunk positions
    let current_offset = 0;

    for (let i = 0; i < total_chunks; i++) {
      // Convert linear index to n-dimensional chunk coordinate
      const inner_chunk_coord = this.#linearToCoord(i, this.#index_shape);
      const key = inner_chunk_coord.join(",");

      if (this.#chunks.has(key)) {
        // Encode the chunk
        const chunk = this.#chunks.get(key)!;
        const chunk_bytes = await this.#inner_codec.encode(chunk);

        // Add to index
        index_entries.push(BigInt(current_offset));
        index_entries.push(BigInt(chunk_bytes.length));

        // Add to data
        chunks_data.push(chunk_bytes);
        current_offset += chunk_bytes.length;
      } else {
        // Fill value chunk
        index_entries.push(MAX_BIG_UINT);
        index_entries.push(MAX_BIG_UINT);
      }
    }

    // Build index
    const index_data = new BigUint64Array(index_entries);
    const index_chunk: Chunk<"uint64"> = {
      data: index_data,
      shape: [...this.#index_shape, 2],
      stride: get_strides([...this.#index_shape, 2], "C"),
    };
    const index_bytes = await this.#index_codec.encode(index_chunk);

    // Combine all data + index
    const total_data_size = chunks_data.reduce(
      (sum, data) => sum + data.length,
      0
    );
    const shard_bytes = new Uint8Array(total_data_size + index_bytes.length);

    let offset = 0;
    for (const chunk_data of chunks_data) {
      shard_bytes.set(chunk_data, offset);
      offset += chunk_data.length;
    }
    shard_bytes.set(index_bytes, offset);

    return shard_bytes;
  }

  /**
   * Convert linear index to n-dimensional coordinate.
   */
  #linearToCoord(linear_index: number, shape: number[]): number[] {
    const coord = [];
    let remaining = linear_index;

    for (let i = shape.length - 1; i >= 0; i--) {
      coord[i] = remaining % shape[i];
      remaining = Math.floor(remaining / shape[i]);
    }

    return coord;
  }
}

import type { Mutable } from "@zarrita/storage";

import { type Array, get_context } from "../hierarchy.js";
import type { Chunk, DataType, Scalar, TypedArray } from "../metadata.js";
import { BasicIndexer, type IndexerProjection } from "./indexer.js";
import type {
  Indices,
  Prepare,
  SetFromChunk,
  SetOptions,
  SetScalar,
  Slice,
} from "./types.js";
import { create_queue } from "./util.js";
import { ShardBuilder } from "../codecs/sharding.js";
import { get_ctr } from "../util.js";

function flip_indexer_projection(m: IndexerProjection) {
  if (m.to == null) return { from: m.to, to: m.from };
  return { from: m.to, to: m.from };
}

export async function set<Dtype extends DataType, Arr extends Chunk<Dtype>>(
  arr: Array<Dtype, Mutable>,
  selection: (number | Slice | null)[] | null,
  value: Scalar<Dtype> | Arr,
  opts: SetOptions,
  setter: {
    prepare: Prepare<Dtype, Arr>;
    set_scalar: SetScalar<Dtype, Arr>;
    set_from_chunk: SetFromChunk<Dtype, Arr>;
  }
) {
  const context = get_context(arr);

  // Check if this is a sharded array
  if (context.kind === "sharded") {
    return await set_sharded(arr, selection, value, opts, setter);
  } else {
    return await set_regular(arr, selection, value, opts, setter);
  }
}

async function set_regular<Dtype extends DataType, Arr extends Chunk<Dtype>>(
  arr: Array<Dtype, Mutable>,
  selection: (number | Slice | null)[] | null,
  value: Scalar<Dtype> | Arr,
  opts: SetOptions,
  setter: {
    prepare: Prepare<Dtype, Arr>;
    set_scalar: SetScalar<Dtype, Arr>;
    set_from_chunk: SetFromChunk<Dtype, Arr>;
  }
) {
  const context = get_context(arr);
  const indexer = new BasicIndexer({
    selection,
    shape: arr.shape,
    chunk_shape: arr.chunks,
  });

  // We iterate over all chunks which overlap the selection and thus contain data
  // that needs to be replaced. Each chunk is processed in turn, extracting the
  // necessary data from the value array and storing into the chunk array.

  const chunk_size = arr.chunks.reduce((a, b) => a * b, 1);
  const queue = opts.create_queue ? opts.create_queue() : create_queue();

  // N.B., it is an important optimisation that we only visit chunks which overlap
  // the selection. This minimises the number of iterations in the main for loop.
  for (const { chunk_coords, mapping } of indexer) {
    const chunk_selection = mapping.map((i) => i.from);
    const flipped = mapping.map(flip_indexer_projection);
    queue.add(async () => {
      // obtain key for chunk storage
      const chunk_path = arr.resolve(
        context.encode_chunk_key(chunk_coords)
      ).path;

      let chunk_data: TypedArray<Dtype>;
      const chunk_shape = arr.chunks.slice();
      const chunk_stride = context.get_strides(chunk_shape);

      if (is_total_slice(chunk_selection, chunk_shape)) {
        // totally replace
        chunk_data = new context.TypedArray(chunk_size);
        // optimization: we are completely replacing the chunk, so no need
        // to access the exisiting chunk data
        if (typeof value === "object") {
          // Otherwise data just contiguous TypedArray
          const chunk = setter.prepare(
            chunk_data,
            chunk_shape.slice(),
            chunk_stride.slice()
          );
          // @ts-expect-error - Value is not a scalar
          setter.set_from_chunk(chunk, value, flipped);
        } else {
          // @ts-expect-error - Value is a scalar
          chunk_data.fill(value);
        }
      } else {
        // partially replace the contents of this chunk
        chunk_data = await arr.getChunk(chunk_coords).then(({ data }) => data);

        const chunk = setter.prepare(
          chunk_data,
          chunk_shape.slice(),
          chunk_stride.slice()
        );

        // Modify chunk data
        if (typeof value === "object") {
          // @ts-expect-error - Value is not a scalar
          setter.set_from_chunk(chunk, value, flipped);
        } else {
          setter.set_scalar(chunk, chunk_selection, value);
        }
      }
      await arr.store.set(
        chunk_path,
        await context.codec.encode({
          data: chunk_data,
          shape: chunk_shape,
          stride: chunk_stride,
        })
      );
    });
  }
  await queue.onIdle();
}

async function set_sharded<Dtype extends DataType, Arr extends Chunk<Dtype>>(
  arr: Array<Dtype, Mutable>,
  selection: (number | Slice | null)[] | null,
  value: Scalar<Dtype> | Arr,
  opts: SetOptions,
  setter: {
    prepare: Prepare<Dtype, Arr>;
    set_scalar: SetScalar<Dtype, Arr>;
    set_from_chunk: SetFromChunk<Dtype, Arr>;
  }
) {
  const context = get_context(arr);
  
  // For sharded arrays, we need to work with shard coordinates
  const indexer = new BasicIndexer({
    selection,
    shape: arr.shape,
    chunk_shape: context.chunk_shape, // This is the chunk shape within shards
  });

  // Group chunks by their shard coordinates
  const shardChunks = new Map<string, {
    chunk_coords: number[];
    mapping: IndexerProjection[];
  }[]>();

  const shard_shape = context.shard_shape!;
  const chunk_shape = context.chunk_shape;
  
  // Calculate chunks per shard in each dimension
  const chunks_per_shard = shard_shape.map((s, i) => s / chunk_shape[i]);

  for (const { chunk_coords, mapping } of indexer) {
    // Calculate which shard this chunk belongs to
    const shard_coords = chunk_coords.map((coord, i) => 
      Math.floor(coord / chunks_per_shard[i])
    );
    const shard_key = shard_coords.join(',');
    
    if (!shardChunks.has(shard_key)) {
      shardChunks.set(shard_key, []);
    }
    
    shardChunks.get(shard_key)!.push({ chunk_coords, mapping });
  }

  const queue = opts.create_queue ? opts.create_queue() : create_queue();

  // Process each affected shard
  for (const [shard_key, chunks] of shardChunks) {
    queue.add(async () => {
      const shard_coords = shard_key.split(',').map(Number);
      
      // Create ShardBuilder for this shard
      // We need to extract the actual data type string from the generic
      const sample_array = new context.TypedArray(1);
      let data_type: Dtype;
      if (sample_array instanceof Int8Array) data_type = "int8" as Dtype;
      else if (sample_array instanceof Uint8Array) data_type = "uint8" as Dtype;
      else if (sample_array instanceof Int16Array) data_type = "int16" as Dtype;
      else if (sample_array instanceof Uint16Array) data_type = "uint16" as Dtype;
      else if (sample_array instanceof Int32Array) data_type = "int32" as Dtype;
      else if (sample_array instanceof Uint32Array) data_type = "uint32" as Dtype;
      else if (sample_array instanceof Float32Array) data_type = "float32" as Dtype;
      else if (sample_array instanceof Float64Array) data_type = "float64" as Dtype;
      else if (sample_array instanceof BigInt64Array) data_type = "int64" as Dtype;
      else if (sample_array instanceof BigUint64Array) data_type = "uint64" as Dtype;
      else throw new Error(`Unsupported data type for array: ${sample_array.constructor.name}`);
      
      const builder = new ShardBuilder(
        shard_shape,
        context.sharding_config!,
        data_type,
        context.fill_value
      );

      // Process each chunk in this shard
      for (const { chunk_coords, mapping } of chunks) {
        // Calculate chunk coordinates within the shard
        const chunk_in_shard_coords = chunk_coords.map((coord, i) => 
          coord % chunks_per_shard[i]
        );
        
        const chunk_selection = mapping.map((proj) => proj.from);
        const flipped = mapping.map(flip_indexer_projection);
        
        let chunk_data: TypedArray<Dtype>;
        const chunk_size = chunk_shape.reduce((a, b) => a * b, 1);
        const chunk_stride = context.get_strides(chunk_shape);

        if (is_total_slice(chunk_selection, chunk_shape)) {
          // Completely replace the chunk
          chunk_data = new context.TypedArray(chunk_size);
          
          if (typeof value === "object") {
            const chunk = setter.prepare(
              chunk_data,
              chunk_shape.slice(),
              chunk_stride.slice()
            );
            // @ts-expect-error - Value is not a scalar
            setter.set_from_chunk(chunk, value, flipped);
          } else {
            // @ts-expect-error - Value is a scalar
            chunk_data.fill(value);
          }
        } else {
          // Partially replace the chunk - need to read existing data first
          try {
            const existing_chunk = await arr.getChunk(chunk_coords);
            chunk_data = existing_chunk.data;
          } catch (error) {
            // Chunk doesn't exist yet, create with fill value
            chunk_data = new context.TypedArray(chunk_size);
            (chunk_data as any).fill(context.fill_value);
          }

          const chunk = setter.prepare(
            chunk_data,
            chunk_shape.slice(),
            chunk_stride.slice()
          );

          if (typeof value === "object") {
            // @ts-expect-error - Value is not a scalar
            setter.set_from_chunk(chunk, value, flipped);
          } else {
            setter.set_scalar(chunk, chunk_selection, value);
          }
        }

        // Add this chunk to the shard builder
        const chunk_obj: Chunk<Dtype> = {
          data: chunk_data,
          shape: chunk_shape.slice(),
          stride: chunk_stride.slice(),
        };
        builder.setChunk(chunk_in_shard_coords, chunk_obj);
      }

      // Build the complete shard
      const shard_data = await builder.build();
      
      // Store the shard using the shard key encoding
      const shard_path = arr.resolve(
        context.encode_chunk_key(shard_coords)
      ).path;
      
      await arr.store.set(shard_path, shard_data);
    });
  }

  await queue.onIdle();
}

function is_total_slice(
  selection: (number | Indices)[],
  shape: readonly number[]
): selection is Indices[] {
  // all items are Indices and every slice is complete
  return selection.every((s, i) => {
    // can't be a full selection
    if (typeof s === "number") return false;
    // explicit complete slice
    const [start, stop, step] = s;
    return stop - start === shape[i] && step === 1;
  });
}

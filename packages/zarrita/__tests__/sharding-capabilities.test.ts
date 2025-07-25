import { expect, it } from "vitest";
import { ShardBuilder } from "../src/codecs/sharding.js";
import { get_strides } from "../src/util.js";

it("demonstrates current sharding write capabilities", async () => {
  // 1. We can build shards using the ShardBuilder
  const shard_shape = [2, 2];
  const sharding_config = {
    chunk_shape: [1, 1],
    codecs: [
      {
        name: "bytes",
        configuration: { endian: "little" },
      },
    ],
    index_codecs: [
      {
        name: "bytes",
        configuration: { endian: "little" },
      },
    ],
  };

  const builder = new ShardBuilder(shard_shape, sharding_config, "int32", 0);

  // Add chunks to build a complete shard
  builder.setChunk([0, 0], {
    data: new Int32Array([1]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  builder.setChunk([0, 1], {
    data: new Int32Array([2]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  builder.setChunk([1, 0], {
    data: new Int32Array([3]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  builder.setChunk([1, 1], {
    data: new Int32Array([4]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  // 2. We can successfully build a complete shard
  const shard_bytes = await builder.build();
  expect(shard_bytes).toBeInstanceOf(Uint8Array);
  expect(shard_bytes.length).toBeGreaterThan(0);

  // 3. The shard contains data + index as per zarr v3 specification
  // (Exact verification would require implementing a shard reader)
  expect(shard_bytes.length).toBeGreaterThan(16); // At least some data + index
});

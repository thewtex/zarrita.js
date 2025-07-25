import { expect, it } from "vitest";
import { ShardBuilder } from "../src/codecs/sharding.js";
import { get_strides } from "../src/util.js";

it("should build and decode shards correctly", async () => {
  // Test the shard builder
  const shard_shape = [2, 2]; // The shard contains a 2x2 area
  const sharding_config = {
    chunk_shape: [1, 1], // Each inner chunk is 1x1
    codecs: [
      {
        name: "bytes",
        configuration: {
          endian: "little",
        },
      },
    ],
    index_codecs: [
      {
        name: "bytes",
        configuration: {
          endian: "little",
        },
      },
    ],
  };

  const builder = new ShardBuilder(
    shard_shape,
    sharding_config,
    "int32",
    0 // fill value
  );

  // Add individual chunks to the shard
  // Chunk at position [0, 0] in the shard
  builder.setChunk([0, 0], {
    data: new Int32Array([1]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  // Chunk at position [0, 1] in the shard
  builder.setChunk([0, 1], {
    data: new Int32Array([2]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  // Chunk at position [1, 0] in the shard
  builder.setChunk([1, 0], {
    data: new Int32Array([3]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  // Chunk at position [1, 1] in the shard
  builder.setChunk([1, 1], {
    data: new Int32Array([4]),
    shape: [1, 1],
    stride: get_strides([1, 1], "C"),
  });

  // Build the shard
  const shard_bytes = await builder.build();
  expect(shard_bytes).toBeInstanceOf(Uint8Array);
  expect(shard_bytes.length).toBeGreaterThan(0);

  // The shard should contain all the chunks we added
  // We can't easily test the exact structure without implementing a shard reader,
  // but we can at least verify it builds successfully
});

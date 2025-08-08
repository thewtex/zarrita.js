import { expect, it } from "vitest";
import * as zarr from "../src/index.js";
import { get_context } from "../src/hierarchy.js";

// Mock store that supports range requests
class MockStore {
  private data = new Map<string, Uint8Array>();

  async get(key: string): Promise<Uint8Array | undefined> {
    return this.data.get(key);
  }

  async set(key: string, value: Uint8Array): Promise<void> {
    this.data.set(key, value);
  }

  async delete(key: string): Promise<void> {
    this.data.delete(key);
  }

  keys(): AsyncIterator<string> {
    const keys = Array.from(this.data.keys());
    let index = 0;
    const iterator = {
      next: () =>
        Promise.resolve({
          value: keys[index++],
          done: index > keys.length,
        }),
    };
    return iterator as AsyncIterator<string>;
  }

  // Add range request support
  async getRange(
    key: string,
    options: { offset?: number; length?: number; suffixLength?: number }
  ): Promise<Uint8Array | undefined> {
    const data = this.data.get(key);
    if (!data) return undefined;

    if (options.suffixLength !== undefined) {
      return data.slice(-options.suffixLength);
    }

    if (options.offset !== undefined && options.length !== undefined) {
      return data.slice(options.offset, options.offset + options.length);
    }

    return data;
  }

  readonly path?: string = undefined;
}

it("should create and write to sharded arrays end-to-end", async () => {
  const store = new MockStore();

  // Create a sharded array
  const arr = await zarr.create(store, {
    shape: [4, 4],
    chunk_shape: [2, 2], // This becomes the shard shape
    data_type: "int32",
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [1, 1], // Inner chunk shape within each shard
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
        },
      },
    ],
  });

  // Verify it's a sharded array
  const context = get_context(arr);
  expect(context.kind).toBe("sharded");
  expect(context.sharding_config).toBeDefined();
  expect(context.shard_shape).toEqual([2, 2]);

  // Write some data to the array - just write a single value for now
  await zarr.set(arr, null, 42);

  // Verify we can read the data back
  const result = await zarr.get(arr, [null, null]);
  const expected = new Int32Array(16).fill(42);
  expect(result.data).toEqual(expected);
  expect(result.shape).toEqual([4, 4]);
});

it("should write partial values to sharded arrays", async () => {
  const store = new MockStore();

  // Create a sharded array
  const arr = await zarr.create(store, {
    shape: [4, 4],
    chunk_shape: [2, 2], // This becomes the shard shape
    data_type: "int32",
    fill_value: 0,
    codecs: [
      {
        name: "sharding_indexed",
        configuration: {
          chunk_shape: [1, 1], // Inner chunk shape within each shard
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
        },
      },
    ],
  });

  // Write to a specific single position
  await zarr.set(arr, [1, 1], 100);

  // Read back and verify
  const result = await zarr.get(arr, [null, null]);
  expect(result.data[1 * 4 + 1]).toBe(100); // position [1,1] in flattened array

  // Check that other positions are still fill value
  expect(result.data[0]).toBe(0);
  expect(result.data[1]).toBe(0);
  expect(result.data[4]).toBe(0);
});

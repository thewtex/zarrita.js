# Sharding with zarr v3

Sharding is a performance optimization in zarr v3 that groups multiple small chunks into larger "shard" files. This reduces the number of files and improves I/O performance, especially when dealing with many small chunks.

## Creating a sharded array

```javascript
import * as zarr from "zarrita";
import { FileSystemStore } from "@zarrita/storage";

const store = new FileSystemStore("./data");

// Create a sharded array
const arr = await zarr.create(store, {
  shape: [100, 100],           // Total array shape
  chunk_shape: [10, 10],       // Shard shape (groups of chunks)
  data_type: "float32",
  codecs: [
    {
      name: "sharding_indexed",
      configuration: {
        chunk_shape: [5, 5],     // Inner chunk shape within each shard
        codecs: [                // Codecs applied to each inner chunk
          {
            name: "bytes",
            configuration: { endian: "little" }
          },
          {
            name: "gzip",
            configuration: { level: 5 }
          }
        ],
        index_codecs: [          // Codecs applied to the shard index
          {
            name: "bytes", 
            configuration: { endian: "little" }
          },
          {
            name: "crc32c"
          }
        ]
      }
    }
  ]
});
```

## Understanding shard structure

In the example above:
- The array is 100×100 elements
- Each **shard** is 10×10 elements (from `chunk_shape`)
- Within each shard, there are **inner chunks** of 5×5 elements
- So each shard contains 4 inner chunks (2×2 arrangement of 5×5 chunks)
- The entire array contains 100 shards (10×10 arrangement of 10×10 shards)

## Reading from sharded arrays

Reading works the same as with regular arrays:

```javascript
// Read the entire array
const data = await zarr.get(arr);

// Read a slice
const slice = await zarr.get(arr, [
  zarr.slice(0, 50),    // First 50 rows
  zarr.slice(0, 50)     // First 50 columns
]);

// Read individual chunks (returns inner chunk, not shard)
const chunk = await arr.getChunk([0, 0]); // Gets 5×5 inner chunk
```

## Writing to sharded arrays

> **Note**: Writing to sharded arrays is currently limited in zarrita.js. 
> Full shard building support is under development.

For now, you can read existing sharded arrays created by other zarr implementations:

```javascript
// Read a sharded array created by zarr-python
const store = new zarr.FetchStore("https://example.com/data.zarr");
const arr = await zarr.open.v3(store, { kind: "array" });
const data = await zarr.get(arr);
```

## Benefits of sharding

1. **Fewer files**: Instead of one file per chunk, you get one file per shard
2. **Better I/O**: Larger files are more efficient for most storage systems
3. **Reduced metadata**: Less overhead from file system metadata
4. **Compression efficiency**: Larger chunks often compress better

## When to use sharding

Sharding is most beneficial when:
- You have many small chunks (< 1MB each)
- Your storage system has high per-file overhead
- You're working with time series or other data with natural groupings
- You want to reduce the total number of files in your dataset

## Configuration guidelines

- **Shard shape**: Should be large enough to group multiple chunks (1-100MB)
- **Inner chunk shape**: Should match your typical access patterns
- **Codecs**: Apply compression at the inner chunk level for best results
- **Index codecs**: Usually just bytes + optional checksum

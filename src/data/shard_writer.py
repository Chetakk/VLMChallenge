from pathlib import Path
import json
import webdataset as wds


def write_shards(clips, shard_dir, shard_size=100):

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    sample_index = 0

    sink = None

    for i, clip in enumerate(clips):

        if sample_index % shard_size == 0:
            if sink:
                sink.close()

            shard_path = shard_dir / f"shard_{shard_index:03d}.tar"
            sink = wds.TarWriter(str(shard_path))
            shard_index += 1

        key = f"{clip['clip_id']}"

        target_json = json.dumps({
            "dominant_operation": clip["dominant_operation"],
            "temporal_segment": clip["temporal_segment"],
            "anticipated_next_operation": clip["anticipated_next_operation"]
        })

        sample = {
            "__key__": key,
            "json": target_json,
        }

        for idx, img_path in enumerate(clip.get("image_paths", [])):
            sample[f"image{idx}.txt"] = img_path  # placeholder until RGB exists
            #sample[f"image{idx}.jpg"] = open(img_path, "rb").read() for real data

        sink.write(sample)
        sample_index += 1

    if sink:
        sink.close()

    print(f"Created {shard_index} shard(s) in {shard_dir}")


# from annotation_parser import load_operations
# from clip_builder import build_clips
# # from shard_writer import write_shards

# ops = load_operations("U0101/annotation/openpack-operations/S0100.csv")
# clips = build_clips(ops, "U0101", "S0100")

# # Add dummy image paths (since RGB not available)
# for clip in clips:
#     clip["image_paths"] = [
#         f"U0101/S0100/frame_{frame:06d}.jpg"
#         for frame in clip["sampled_frames"]
#     ]

# write_shards(clips, "data/shards", shard_size=50)
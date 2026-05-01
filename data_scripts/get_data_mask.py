import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import cv2
from tqdm import tqdm
import h5py
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Manager

BAD_MODELS = [
    # mismatched transformation matrices and images
    "07439544cfe342ced9fbf5cb6184b14e6d099d00edff8e4e212b8b86dcb7c121",
    "1a8066e07d8ddadb41b8a5dd7fd9519e6375c3023d2546b3fd3bf99878340ef6",
    "1bd8b664e906ab3b3081a9e0adc03fb5a26676d43b21fc6132547df33f80d15a",
    "37a844db239656c4770e61f67b80a27c37fbb3447224f14a565529d8ec3c06d6",
    "42fb6874cad7fc56efa95d2ec6e8c62787a1e3a324e9d578386643bdb01b3c6d",
    "4c5a9dbac5841ad7cb9a9a036deb6ee86588e6fc59aea5be71aac5bece01c2f0",
    "9bb816bd97d2f9282dcc972580e470e631061aa5d98812ab92bcb8087e9393ee",
    "9ccc61b13d3a9cd51d5dd7eb4e8d5ab866136d10a11055bd22b6a99ef91daa12",
    "a03c2b5a49e81cd6a23713f6c25f43113a899a412e3c5ce2293c291006ce444f",
    "d9b6376623741313bf6da6bf4cdb9828be614a2ce9390ceb3f31cd535d661a75",
    "e1a24d2d91af6349ca7b52bd5f4216d8000caad93fb07d413ecb04d5c5d657d4",
    "fff0ff507a7d886e7c843949050a7b3f7f8644d956f2029316b749c6625d1b1b",
    # no image
    "237d157619074ffa91647c06e229b75bd26ba0081c0df4dc8af20fe0372b1b1f",
    "f27683b253414c3f0ba0166068170aa3fa10ea06362d3ad00a8bbd6470f7c87a",
    "e0b146d7b1c665abe8851017e30bec17cb95313a88411f9d49320bc40cd271b9",
    "debb5f7949037981bef217158b35222723f2962f0efdc751864feddbacb2582d",
    "fa3adec6c5dba35e12ab2ec29344b70e4b4092aec0c1d90d9651e61b5ef98f68",
    "e0481c9c7214be3d99e7c06228ccf580a80b5182bbbde8f52a5b263fa653ff9f",
    "e3becf031badc0d41aaaa8826527be8e2571978526199b5ab060969adf838e78",
    "f54186955ab849c68dcc8e7c01afea1f9a663384ccfe1e3f8dc95f917c8c8c14",
    "d27b23eb5a5b6d45ca584d2b4c73698fc0623554b8379e300b4b08825683daa8",
    "ec0b164a1e5d2b013899856ab8baae5be8496d78a15286fb4373744b53192e50",
    "f89afc7b2b7ac787e1b7345f0d2664f308f743214c1d3c24947a3e95362de5b7",
    "14618d5c5037924036a723a2e4d6a00d24808114a768a3d7eab3da073acc54e9",
    "f8ee8358cb568c92aeb33c65c3ad7a9e7c8ff8e1063a8dc42584b24feedab2cb",
    "e6074624334f41cced25cb54f5110f7d2fe49a9de89808865074d88769577116",
    "d63335eb5192e66cb6a1b51f11f2235dc4c9530564e4bb9dd064fa03d0bd3d6b",
    "e15bbe99274711df0cfabffbff5ea739881878b7ccc2b11b92fae016b993bfab",
    "f962675947e84438029244b40810e6110c36db161b598165b7da5a9bdc79261d",
    "1b9fe38bcd6ab6d904c7f6fa49255996576e74ac95f933dcd5466a0c002002a9",
    "02dbeaac48195aa9dce4764dd453ccbe0c8ddf2612c9b9ec20dec7ffc3f06093",
    "ed16328235c610f15405ff08711eaf15d88a0503884f3a9ccb5a0ee69cb4acb5",
    "42acb74eae53058af0c1e3d1882d0ddcf10993db78f119c9ec48c3ac0418fdf0",
    "ed99086838cd554bbd9ea884939fef0144e3704b9ed68c59617cca3c183710c5",
    "1f01cfe937cc98173e8b609dd94423282557ff585d5bb59fd5017bf6a463c3a3",
    "eaae1f12cf872c9a7aae8e640bd57f6d0d6814917c8681a64948fdc04e93a291",
    "0ad50d6a12d5ddb763027b0608aa18490896e16ac13a1ec93ea1f9fddec9dda1",
    "fd86362a965ad165d1ad18873e1027392bbb1ba62daeada466a00e3dcc818f40",
    "db9742f75d7f247ec7ed36c20bc4b96194ffa1892515a95ce6985d7edd90d496",
    "d157bfce01f6e7dd824e337acd577e3b95409f616b8c8a45aa98d64b353ba534",
    "2d7efce8f7585dde4f46583ef9480e4caca6ea9166e59c42953668a9f4ef50d5",
    "d54d78994a4014ed0f9c1cd477bd5e6c67d84c6288b3a38ab2bd1488894ac90a",
    "2c33c445aee6e8a30a8963e0daee67e507ce23f71a431d7224722c4332f4846d",
    "dd9406d83300847a2175bcdc60cb0f2ad8edb906cf63f5ab85e05283258de1c6",
    "f6aeac8393ff9f01478014d02694ebb94eb9be847abb7c00b03cd4ffaf591685",
    "e4b0f1da8edec62dbe256444ece29546fb3e021d3ae643b643964646a1a3c0e6",
    "0b13ac82638ead088e4b724f20954c4fba5221b8e4da1c97df3204a31a2b5a12",
    "efbb218e4a439ee531b0408f2510c14beb2092e1f58218bb815ba532eeca70d8",
    "da89f5de918896a088480d6944d28339064a4e6c4507dce89277b37d95927833",
    "334c33421b69017d7a9550fcadaf943a6ec021611b19820260f3ea17c7f4e941",
    "e78f8cebd2bd93d960bfaeac18fac0bb2524f15c44288903cd20b73e599e8a81",
    "41976b976a156d2744b3ff760ea1385b3d1dd53d002ecb1ac4b2953515d5c30e",
    "fb18146bac923b4fb3482b76f037e34013816f0e014bbd87721e652c63300689",
    "f42ba19da9b9aeb127fc8745189a9bcca3756a9404d17e9d79355d63b9ba698c",
    "103d51e1876debfbe40b5b55ef0aac6622b4b414937baf7ac4c77f49703f0350",
    "f7f57da984ab57a34986b24f324d78e868f47288c084d13f9d1dff3c77b083fd",
    "fb4bfdc47baccc5e69a6e4b2fd990d75a25424abed7684f3537f404a49ee2762",
    "28cf5e213a7ed1186fa7ecd8bd80f34fdc2b52b195dae41ed6d1c09067c34484",
    "f71ac346cd0fc4652a89afb37044887ec3907d37d01d1ceb0ad28e1a780d8e03",
    "fafa9530c7d75259c101c96d0f86bd78ddf5ad3c5dee1649e772f6daefae5d00",
    "f0da46dc0a419370aa8d68ec5a531ba765cb3be1085d8736dd72d3fb4be3e106",
    "3881e22f7c0805e9ec0c6efb140b82e965db488f49eeb8ebbe2991bd76a85aba",
    "3b62589d0a88b4caa213f2bf8055aa8c6d594565242cd681aea6c5cfeb22b88f",
    "e04f626fa13508fa61778a8eea64672dfc42da6791326d1e9dbbcb5af08ff8b4",
    "dafa9c7cbda9d1ddaa8a2b51fc8c54f4eb44161f5e5c53685dc744580ca77751",
    "21871afb4e962cbd8c7e6d339a171adc43bf366a8a018358b4658e9967beb5d4",
    "faca8d34cb1bb53ef1f5522b1d1467fc17c63be821bb0b85f88e74bb017e8b17",
    "ec8037cde5de9dc3b6b2b13b831b6f9750e8b5b8208537a33c253246954c6d56",
    "edb4efeec01a13869cc06bb9e9c45b94f33c857dde39d66973ac2d86555d12e7",
    "fef478650bf33defe96b866d02555d6b3c6d7e3fbf89f08ebdd353fc01b9e0aa",
    "eb572604667f32be5e0fe093cb8247d0cbc9c636345c63e1f464d42fdef6e5c9",
    "e57e2c89a8ee491da2229175658b9f3eb3ffc536511a6b03596730e79def9895",
    "e0f529f24e1a4ae0f8188e66df633ffafff67a629e1221630da0aa0866248761",
    "fe6120f6d13fcdfa10cce10d7b0dc3ff8bb99b9b0293e67f6693b86a3cdda6da",
    "006d25fe651fe92558a5a28e1911dbd6fde1679491ce3d6ddefa4d2ae9eb9ae5",
    "2713ce42632878c2d125993057e6ee613ed0fd3d4a16f25f5cee07be19e66f58",
    "e2591fdd5af063ec8e61ec3f27d9c1a81fb6ef3312760ed5417909a1ff0c5349",
    "215c2e1f0772d742e0cc088a24651dabb8922ae9e00cbe2cc77a7f817bc75928",
    "d008b7daa291deb081aadf8e2ee915382080056d61be67b2d16f2d590a46aac8",
    "f1af570cb06b582e21332a7c24a7a15b48ff41c2fcc2f2fd52484db4061193f1",
    "ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101",
    "e84da23da065ee9f2b20968bbbc54af110873903018b00a8729d34b823364ab1",
    "faf1889500a26213810cd28c4228ed7f967c2cf4671bacd6b14b2d2f97889965",
    "93435c6190069a48398d94b545b615efbc11cbfb18ae824f197ddcde2233b442",
    "5df2cbdfeb07629e539ad7393bd0a94c39bfa1271b9d49bdb3f62c215a7891f0",
    "b2ddf3abc829eb768845d960e00b69b9bca1bc216fc1402590f969a231460e92",
]


def pre_select_bounding_boxes(image_path, mask_generator, clip_index, top_k, do_vis=False):
    """
    Pre-select bounding boxes for the given image using SAM2.
    Select Top k objects in the roughly middle part of the image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    if len(masks) == 0:
        return None, None
    image_width, image_height = image.shape[1], image.shape[0]

    # filtering masks that meet the requirements
    new_masks = []
    for mask in masks:
        if mask["area"] < 200 or mask["area"] > 20000:
            continue
        if mask["point_coords"][0][0] < image_width / 4 or mask["point_coords"][0][0] > image_width * 3 / 4 \
                or mask["point_coords"][0][1] < image_height / 4 or mask["point_coords"][0][1] > image_height * 3 / 4:
            continue
        new_masks.append(mask)

    new_masks.sort(key=lambda x: x["predicted_iou"], reverse=True)
    new_masks = new_masks[:top_k]

    if len(new_masks) == 0:
        return None, None

    prompt_points = [mask["point_coords"][0] for mask in new_masks]
    prompt_points = np.array(prompt_points)

    bounding_boxes = [
        np.array([
            mask["bbox"][0], 
            mask["bbox"][1], 
            mask["bbox"][0] + mask["bbox"][2], 
            mask["bbox"][1] + mask["bbox"][3]
        ]) # XYWH -> X1Y1X2Y2
        for mask in new_masks
    ]
    bounding_boxes = np.array(bounding_boxes)

    assert len(bounding_boxes) == len(prompt_points), "Length of boxes and points must be the same"


    if do_vis:
        # Visualization with segmentation masks
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        colors = [np.random.randint(0, 255, 3).tolist() for _ in new_masks] # color for each mask
        overlay = vis_image.copy()
        alpha = 0.5  # transparency
        for mask, color in zip(new_masks, colors):
            overlay[mask["segmentation"]] = color
            # 绘制分割区域
            contours, _ = cv2.findContours(
                mask["segmentation"].astype(np.uint8), 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, -1)  # -1表示填充
        vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)

        for box in bounding_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        for point in prompt_points:
            x, y = point.astype(int)
            cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
        
        # Save visualization
        vis_dir = Path("./mask_visualization")
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{image_path.parents[1].name}_{clip_index}.jpg"), vis_image)
    

    return bounding_boxes, prompt_points


def track_objects_in_clip(video_path, initial_boxes, prompt_points, predictor, clip_index, do_vis, img_size, inference_frame_index, top_k):
    """
    return object mask that exist throughout the video clip
    """
    all_masks = []
    valid_objects = [True] * len(initial_boxes)  # keep track of valid objects

    inference_state = predictor.init_state(video_path)
    predictor.reset_state(inference_state)
    
    obj_id = 0
    for point, box in zip(prompt_points, initial_boxes):
        _  = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=inference_frame_index,
            obj_id=obj_id,
            points=np.array([point]),
            labels=np.array([1], np.int32),
            box=np.array([box]),
        )
        obj_id += 1

    video_segments = []
    prev_masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments.append([])
        for i, out_obj_id in enumerate(out_obj_ids):
            # if the mask is already invalid, just skip and don't do the calcualation
            if not valid_objects[out_obj_id]:
                video_segments[out_frame_idx].append(np.zeros(img_size, dtype=bool))
                continue
            
            instance_mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]

            # Calculate IOU with previous frame's mask
            if out_obj_id in prev_masks:
                prev_mask = prev_masks[out_obj_id]
                intersection = np.logical_and(instance_mask, prev_mask).sum()
                union = np.logical_or(instance_mask, prev_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                
                # Example: Mark invalid if IOU drops below threshold
                if iou < 0.15:  # Adjust threshold as needed
                    valid_objects[out_obj_id] = False
            prev_masks[out_obj_id] = instance_mask.copy()

            # if the instance disapper in one of the frame, 
            # set the valid flag to False
            if instance_mask.sum() == 0.0:
                # print(f"frame {out_frame_idx}, out_obj_id {out_obj_id}")
                valid_objects[out_obj_id] = False
            video_segments[out_frame_idx].append(instance_mask)
        
        video_segments[out_frame_idx] = np.stack(video_segments[out_frame_idx])

    video_segments = np.stack(video_segments)
    video_segments = video_segments[:, np.array(valid_objects), :, :]
    video_segments = video_segments[:, :top_k, :, :]

    if do_vis:
        # initialize video writers
        frame_len = len(video_path)
        object_num = video_segments.shape[1]
        color_to_save = [np.random.randint(0, 255, 3).tolist() for _ in range(object_num)]
        save_path = Path("./mask_visualization") / f"{video_path[0].parents[1].name}_{clip_index}.mp4"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        first_frame = cv2.imread(str(video_path[0]))
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(save_path), fourcc, 5.0, (width, height))
        for frame_id in range(frame_len):
            frame = cv2.imread(str(video_path[frame_id]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame.copy()
            # object mask
            for obj_id in range(object_num):
                instance_mask = video_segments[frame_id, obj_id]
                # fill the masked region
                overlay[instance_mask] = color_to_save[obj_id]
                # draw the contours
                contours, _ = cv2.findContours(
                    instance_mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)
            alpha = 0.5
            visualized = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            video_frame = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
            video_writer.write(video_frame)
        video_writer.release()

    return video_segments

def writer_process(filename, queue):
    """Writer process that write masks into h5py to avoid conflict"""
    with h5py.File(filename, 'r+') as f:
        while True:
            msg = queue.get()
            if msg is None:  # end
                break
            model_id, all_object_masks_per_model, frame_validate = msg
            group = f.create_group(model_id)
            group.create_dataset('masks', data=all_object_masks_per_model, dtype=np.uint8)
            group.create_dataset("frame_validate", data=frame_validate, dtype=np.bool)


def mask_generating_worker(gpu_id, model_chunk, writing_queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    top_k = args.top_k
    do_vis = args.do_vis
    clip_len = args.clip_len
    inference_frame_index = 0
    img_size = (270, 480)
    
    # intialize
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    total_available_instance_num = 0
    
    os.makedirs("./mask_gen_log", exist_ok=True)

    iterable = model_chunk
    if gpu_id == available_gpus[0]:
        iterable = tqdm(model_chunk, desc=f"GPU{gpu_id}", file=sys.stdout, dynamic_ncols=True)

    for model in iterable:
        model_id = model.name
        if model_id in BAD_MODELS:
            continue
        with open(f"./mask_gen_log/gpu_{gpu_id}.txt", "a", encoding="utf-8") as f:
            f.write(f"Starting to process: {model}\n")
        all_object_masks_per_model = []
        video_path = model / "images_4"
        frame_names = [
            p for p in os.listdir(video_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]

        target_size = (img_size[1], img_size[0])
        with Image.open(video_path / frame_names[0]) as img:
            needs_resize = img.size != target_size
        if needs_resize:
            with open(f"./mask_gen_log/gpu_{gpu_id}.txt", "a", encoding="utf-8") as f:
                f.write(f"Resizing {model} frames to {target_size}\n")
            for frame_name in frame_names:
                frame_path = video_path / frame_name
                with Image.open(frame_path) as img:
                    if img.size == target_size:
                        continue
                    resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized.save(frame_path)

        # sort the image list according to frame numebr
        frame_names.sort(key=lambda p: int(re.search(r'\d+', p).group()))
        # divided_frames = np.array_split(np.array(frame_names), 9)
        divided_frames = [frame_names[i:i+clip_len] for i in range(0, len(frame_names), clip_len)]

        frame_validate = []
        for i, video_clip in enumerate(divided_frames):
            # get object candidate's bounding boxes based on the first frame
            bounding_boxes, prompt_points = pre_select_bounding_boxes(
                video_path / video_clip[inference_frame_index], 
                mask_generator, 
                clip_index=i,
                top_k=top_k*2,
                do_vis=do_vis,
            )
            if bounding_boxes is None:
                # append zero padding and set cooresponding frame to false
                all_object_masks_per_model.append(np.zeros((len(video_clip), *img_size), dtype=np.int8))
                frame_validate.extend([False] * len(video_clip))
                continue

            frame_paths = [video_path / name for name in video_clip]
            object_masks = track_objects_in_clip(
                frame_paths, 
                bounding_boxes, 
                prompt_points, 
                video_predictor, 
                clip_index=i,
                do_vis=do_vis, 
                img_size=img_size,
                inference_frame_index=inference_frame_index,
                top_k=top_k,
            )

            if object_masks.sum() == 0:
                all_object_masks_per_model.append(np.zeros((len(video_clip), *img_size), dtype=np.int8))
                frame_validate.extend([False] * len(video_clip))
                continue

            # transform object mask to bitmap 
            T, num_objects, H, W = object_masks.shape
            total_available_instance_num += num_objects
            bitmap = np.zeros((T, H, W), dtype=np.uint8)
            for t in range(T):
                for obj_id in range(num_objects):
                    bitmap[t][object_masks[t, obj_id]] += 2 ** obj_id 
            object_masks = bitmap

            all_object_masks_per_model.append(object_masks)
            frame_validate.extend([True] * len(video_clip))
        
        all_object_masks_per_model = np.concat(all_object_masks_per_model, axis=0)
        frame_validate = np.array(frame_validate)

        # lock = FileLock(Path(mask_cache_path).name + ".lock")

        # with lock:
        #     with h5py.File(mask_cache_path, "r+") as f:
        #         group = f.create_group(model_id)
        #         group.create_dataset('masks', data=all_object_masks_per_model, dtype=np.uint8)
        #         group.create_dataset("frame_validate", data=frame_validate, dtype=np.bool)
        writing_queue.put((model_id, all_object_masks_per_model, frame_validate))

    return (total_available_instance_num, gpu_id)

def get_args_parser():
    parser = argparse.ArgumentParser("LRM Triplane", add_help=False)

    parser.add_argument("--mask_cache_path", type=str, default="./data/mask_cache.h5")
    parser.add_argument("--data_root", type=str, default="./data/dl3dv_960")
    parser.add_argument("--subset_to_process", type=str, nargs="+", default=["1K"])
    parser.add_argument("--available_gpus", type=int, nargs="+", default=[0,1,2,3,5,6])
    parser.add_argument("--clip_len", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--do_vis", action="store_true", default=False)

    return parser

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    available_gpus = args.available_gpus
    print(f"Available GPU nums: {len(available_gpus)}")
    mask_cache_path = args.mask_cache_path
    if not os.path.exists(mask_cache_path):
        # create the h5 file
        with h5py.File(mask_cache_path, "w") as h5:
            pass

    data_root = Path(args.data_root)
    subset_to_process = args.subset_to_process
    models = []
    for subset in subset_to_process:
        model_ids = os.listdir(data_root / subset)
        models = models + [data_root / subset / model_id for model_id in model_ids]
    print(f"Found {len(models)} models in {len(subset_to_process)} subset")

    unprocessed_models = []
    with h5py.File(mask_cache_path, "r") as f:
        for model in models:
            model_id = model.name
            if model_id not in f:
                unprocessed_models.append(model)
    print(f"Found {len(unprocessed_models)} unprocessed models")
    models = unprocessed_models


    writing_queue = Manager().Queue()
    writer = Process(target=writer_process, args=(mask_cache_path, writing_queue))
    writer.start()

    if not args.do_vis:
        # split to chunk according to GPU num
        chunk_size = len(models) // len(available_gpus) + 1
        model_chunks = [models[i:i+chunk_size] for i in range(0, len(models), chunk_size)]

        # # handle the data preprocession with multi-threading
        futures = []
        with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
            for gpu_id, chunk in zip(available_gpus, model_chunks):
                future = executor.submit(mask_generating_worker, gpu_id, chunk, writing_queue, args)
                futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                tot_num, gpu_id = result
                print(f"Secure result from gpu_id {gpu_id}: tot model num {tot_num}")
            except Exception as e:
                print(f"GPU Process failed: {e}")

        writing_queue.put(None)
        writer.join(writer.join(timeout=30))
    else:
        models = models
        mask_generating_worker(available_gpus[0], models, writing_queue, args)

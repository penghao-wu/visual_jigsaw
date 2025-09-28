import io
import random
import decord
import torch
import numpy as np
from fractions import Fraction
from typing import List, Sequence, Any, Dict
from PIL import Image, ImageDraw, ImageFont

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from qwen_vl_utils.vision_process import smart_resize, ceil_by_factor, floor_by_factor, FRAME_FACTOR, IMAGE_FACTOR

def draw_numbered_markers(rgb, points, numbers=None, radius=25, alpha=180):
    """
    Draw semi-transparent red circle markers with numbers inside.
    
    Args:
        rgb: np.ndarray (H,W,3) or PIL.Image
        points: list of (h,w) coordinates
        numbers: list of ints or strings (same length as points), default 1..N
        radius: circle radius
        alpha: transparency (0=transparent, 255=solid)
    
    Returns:
        PIL.Image with markers drawn
    """
    if isinstance(rgb, np.ndarray):
        img = Image.fromarray(rgb).convert("RGBA")
    else:
        img = rgb.convert("RGBA")

    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    
    if numbers is None:
        numbers = [str(i+1) for i in range(len(points))]

    for (h, w), num in zip(points, numbers):
        x, y = w, h

        # Draw semi-transparent red circle
        draw.ellipse(
            (x-radius, y-radius, x+radius, y+radius),
            fill=(255,0,0,alpha),
            outline=(255,0,0,255),
            width=2
        )

        # Draw number centered
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", radius)  # adjustable font
        except:
            font = ImageFont.load_default()
        # get text size using font
        bbox = font.getbbox(num)  # (left, top, right, bottom)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw/2, y - th/2), num, fill="white", font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")

def mask_random_regions(img, m, n, k, seed=None, chosen_indices=None):
    """
    Divide `img` into an m x n grid. Randomly choose k regions to mask:
      - Fill each chosen region with black.
      - Draw a white inner border entirely within the masked tile.
      - Draw a centered white number (1..k in raster order of the chosen tiles).
    Returns:
      masked_img: PIL.Image after masking
      masked_tiles: list of ORIGINAL (unmasked) subimages for the masked regions,
                    ordered by the numbering shown (i.e., 1..k in raster order).
    """
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers.")
    total = m * n
    if not (0 <= k <= total):
        raise ValueError(f"k must be between 0 and {total} (inclusive).")

    masked_img = img.copy()
    W, H = masked_img.size

    # Grid boundaries with even rounding
    xs = [round(i * W / n) for i in range(n + 1)]
    ys = [round(j * H / m) for j in range(m + 1)]

    # Boxes in raster order (row-major)
    boxes = []
    for r in range(m):
        for c in range(n):
            boxes.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))

    if chosen_indices is None:
        rng = random.Random(seed)
        # Sort to ensure raster order, then we will number 1..k along this ordering
        chosen_indices = sorted(rng.sample(range(total), k))
    else:
        chosen_indices = chosen_indices

    # Save ORIGINAL content of masked tiles in the same numbering order 1..k
    masked_tiles = [img.crop(boxes[idx]) for idx in chosen_indices]

    draw = ImageDraw.Draw(masked_img)

    # Load font
    try:
        from PIL import ImageFont as _ImageFont
        # font_path = os.path.join(os.path.dirname(_ImageFont.__file__), "DejaVuSans.ttf")
        font_path = "DejaVuSans.ttf"
        tile_w = max(1, round(W / n))
        tile_h = max(1, round(H / m))
        font_size = max(10, int(0.4 * min(tile_w, tile_h)))
        font = _ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Border width and inset to keep border fully inside the tile
    tile_w = max(1, round(W / n))
    tile_h = max(1, round(H / m))
    border_w = max(2, int(0.03 * min(tile_w, tile_h)))  # ~3% of tile size
    inset = border_w  # drawing the rectangle inset by >= border_w keeps the stroke inside

    for shown_num, idx in enumerate(chosen_indices, start=1):
        l, t, r, b = boxes[idx]

        # Fill tile black
        draw.rectangle([l, t, r, b], fill=(0, 0, 0))

        # Inner white border fully inside the tile
        il, it, ir, ib = l + inset, t + inset, r - inset, b - inset
        if ir > il and ib > it:
            draw.rectangle([il, it, ir, ib], outline=(255, 255, 255), width=border_w)

        # Centered number (1..k)
        text = str(shown_num)
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
            tw, th = (x1 - x0), (y1 - y0)
        except Exception:
            # Fallback: approximate via font.getbbox
            x0, y0, x1, y1 = font.getbbox(text)
            tw, th = (x1 - x0), (y1 - y0)

        cx = l + (r - l) / 2
        cy = t + (b - t) / 2
        tx = int(cx - tw / 2)
        ty = int(cy - th / 2)

        # Light dark outline for contrast, then white text
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text((tx + dx, ty + dy), text, font=font, fill=(64, 64, 64))
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255))

    return masked_img, masked_tiles

def slice_image(img: Image.Image, n_rows: int, n_cols: int) -> List[Image.Image]:
    """
    Slice an image into an n_rows × n_cols grid.
    
    Tiles are returned in reading order:
        row‑0/col‑0, row‑0/col‑1, …, row‑1/col‑0, …, row‑(n_rows‑1)/col‑(n_cols‑1)
    
    If the width or height does not divide evenly, extra pixels are assigned
    to the *last* column and *last* row so that the overall reconstruction
    is loss‑less.
    
    Args
    ----
    img : PIL.Image
        Input image (will not be modified).
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    
    Returns
    -------
    List[PIL.Image]
        List of tiles in row‑major order.
    """
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive integers.")

    w, h = img.size
    base_w, extra_w = divmod(w, n_cols)
    base_h, extra_h = divmod(h, n_rows)

    # Pre‑compute the x and y boundaries for cropping
    x_coords = [0]
    for col in range(n_cols):
        step = base_w + (1 if col < extra_w else 0)
        x_coords.append(x_coords[-1] + step)

    y_coords = [0]
    for row in range(n_rows):
        step = base_h + (1 if row < extra_h else 0)
        y_coords.append(y_coords[-1] + step)

    # Crop tiles
    tiles: List[Image.Image] = []
    for r in range(n_rows):
        for c in range(n_cols):
            left, upper = x_coords[c], y_coords[r]
            right, lower = x_coords[c + 1], y_coords[r + 1]
            tiles.append(img.crop((left, upper, right, lower)))

    return tiles

def split_and_trim_uniform_indices(N: int, K: int, r: float):
    """
    Returns K half-open [start, end) ranges in the ORIGINAL timeline.
    """
    if not (0 <= r < 0.5):
        raise ValueError("r must satisfy 0 ≤ r < 0.5")

    p, q = Fraction(r).limit_denominator().numerator, Fraction(r).limit_denominator().denominator
    block = K * q
    skip  = N % block
    if K == 1:
        skip = 0
    N2    = N - skip

    L_raw = N2 // K
    trim  = (L_raw * p) // q
    if trim * 2 >= L_raw:
        raise ValueError("r is too large for the resulting clip size")

    base = skip
    ranges = []
    for i in range(K):
        start = base + i * L_raw + trim          # inclusive
        end   = base + (i + 1) * L_raw - trim    # exclusive
        ranges.append((start, end))
    return ranges


def split_clip_spatially(clip_video: torch.Tensor, m: int, n: int,
                                interp_mode: str = "bicubic", align_corners: bool = False):
    """
    clip_video: [T, C, H, W] (uint8 or float)
    m, n: positive ints. Produces exactly m*n tiles, row-major.
    Behavior:
      - If H or W not divisible by m/n, first upsample spatially (one interpolation)
        to the next multiples, then hard-slice evenly.
      - No padding, no linspace, no uneven tile sizes.
    Returns:
      list[Tensor] of length m*n, each [T, C, h_tile, w_tile]
    """
    assert clip_video.ndim == 4, "Expect [T, C, H, W]"
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")

    T, C, H, W = clip_video.shape

    # Target size: next multiples (only upsample if needed)
    Ht = ((H + m - 1) // m) * m
    Wt = ((W + n - 1) // n) * n

    if Ht != H or Wt != W:
        # One interpolation to increase size; keep float for quality
        x = clip_video.float()
        clip_video = torch.nn.functional.interpolate(x, size=(Ht, Wt), mode=interp_mode, align_corners=align_corners)

        T, C, H, W = clip_video.shape

    h_tile = H // m
    w_tile = W // n

    tiles = []
    for i in range(m):
        h0 = i * h_tile
        h1 = (i + 1) * h_tile
        for j in range(n):
            w0 = j * w_tile
            w1 = (j + 1) * w_tile
            tiles.append(clip_video[..., h0:h1, w0:w1])

    return tiles

def process_video_jigsaw(video, max_pixels=None, fps_max_frames=None):
    video_path = video['path']
    if any((name in video_path) for name in [
        'liwei_youtube_videos/videos/youtube_video_2024/ytb_nJ11r1kVt14.mp4',
        'liwei_youtube_videos/videos/youtube_video_2024/ytb_UKLnTkIzsxs.mp4',
        'liwei_youtube_videos/videos/youtube_video_2024/ytb_FwoZBsssEXg.mp4',
        'liwei_youtube_videos/videos/youtube_video_2024/ytb_4s2QqSla2CA.mp4',
        'liwei_youtube_videos/videos/youtube_video_2024/ytb_pWRqmt6EEqw.mp4'
    ]):
        raise AssertionError("invalid video file")

    vr = decord.VideoReader(video_path)

    total_frames, video_fps = len(vr), vr.get_avg_fps()

    jigsaw_clip_num = video.get('jigsaw_clip_num', 4)
    fps = video.get('fps', 2)
    fps_min_frames = video.get('fps_min_frames', 2)
    if fps_max_frames is None:
        fps_max_frames = video.get('fps_max_frames', 16)
    min_pixels = video.get('min_pixels', 128 * 28 * 28)
    if max_pixels is None:
        max_pixels = video.get('max_pixels', 128 * 28 * 28)

    spatial_cfg = video.get('spatial_jigsaw_grid_config', None)
    if spatial_cfg is not None:
        if not (isinstance(spatial_cfg, (list, tuple)) and len(spatial_cfg) == 2):
            raise ValueError("spatial_jigsaw_grid_config must be [m, n]")
        m, n = int(spatial_cfg[0]), int(spatial_cfg[1])
        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive integers")
    else:
        m = n = None

    # Half-open temporal ranges
    clip_ranges = split_and_trim_uniform_indices(total_frames, jigsaw_clip_num, 0.1)

    processed_clip_videos, fps_list = [], []

    for abs_start, abs_end in clip_ranges:
        total_frames_per_clip = abs_end - abs_start  # end exclusive

        # frame-count logic (respecting FRAME_FACTOR)
        min_frames = ceil_by_factor(fps_min_frames, FRAME_FACTOR)
        max_frames = floor_by_factor(min(fps_max_frames, total_frames_per_clip), FRAME_FACTOR)
        nframes = total_frames_per_clip / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = floor_by_factor(int(nframes), FRAME_FACTOR)
        assert FRAME_FACTOR <= nframes <= total_frames_per_clip

        # relative → absolute indices
        rel_idx = torch.linspace(0, total_frames_per_clip - 1, nframes).round().long()
        abs_idx = (rel_idx + abs_start).tolist()

        # decode needed frames
        batch = vr.get_batch(abs_idx).asnumpy()                  # (T, H, W, 3)
        clip_video = torch.from_numpy(batch).permute(0, 3, 1, 2) # [T, C, H, W], uint8
        sample_fps = nframes / max(total_frames_per_clip, 1e-6) * video_fps

        if spatial_cfg is None:
            # Single tile path: do smart resize now (interpolation)
            _, _, h, w = clip_video.shape
            out_h, out_w = smart_resize(
                h, w,
                factor=IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            clip_video = transforms.functional.resize(
                clip_video, [out_h, out_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            processed_clip_videos.append(clip_video)
            fps_list.append(sample_fps)
        else:
            # Split first (no padding), then per-tile smart resize (interpolation)
            tiles = split_clip_spatially(clip_video, m, n)
            for tile in tiles:
                _, _, th, tw = tile.shape
                out_h, out_w = smart_resize(
                    th, tw,
                    factor=IMAGE_FACTOR,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                tile = transforms.functional.resize(
                    tile, [out_h, out_w],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                processed_clip_videos.append(tile)
                fps_list.append(sample_fps)

    return processed_clip_videos, fps_list

def center_crop(img: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    """Center crop a PIL image to (crop_width, crop_height)."""
    w, h = img.size
    left   = (w - crop_width) // 2
    top    = (h - crop_height) // 2
    right  = left + crop_width
    bottom = top + crop_height
    return img.crop((left, top, right, bottom))
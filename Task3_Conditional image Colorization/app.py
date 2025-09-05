import io
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from colorize import (
    compose_rules,
    ensure_gray,
    from_uploaded_file_to_bgr,
    make_mask_from_canvas_rgba,
    magic_wand_mask,
    normalize_and_fit_image,
    save_image_bytes,
    to_display_rgb,
)


st.set_page_config(page_title="Conditional Image Colorization", layout="wide")


@dataclass
class ColorRule:
    id: str
    name: str
    color: Tuple[int, int, int]  # RGB
    intensity: float = 1.0
    feather: int = 8
    enabled: bool = True
    mask: np.ndarray = field(default=None, repr=False)


def init_state():
    if 'rules' not in st.session_state:
        st.session_state.rules: List[ColorRule] = []
    if 'base_bgr' not in st.session_state:
        st.session_state.base_bgr = None
    if 'work_bgr' not in st.session_state:
        st.session_state.work_bgr = None
    if 'gray' not in st.session_state:
        st.session_state.gray = None
    if 'selection_mask' not in st.session_state:
        st.session_state.selection_mask = None
    if 'selection_mode' not in st.session_state:
        st.session_state.selection_mode = 'Brush'


def add_rule(mask: np.ndarray, color_rgb: Tuple[int, int, int], intensity: float, feather: int, enabled: bool = True):
    st.session_state.rules.append(
        ColorRule(
            id=str(uuid.uuid4()),
            name=f"Rule {len(st.session_state.rules)+1}",
            color=color_rgb,
            intensity=float(intensity),
            feather=int(feather),
            enabled=bool(enabled),
            mask=mask.astype(np.float32) if mask is not None else None,
        )
    )


def reorder_rule(idx: int, direction: int):
    # direction: -1 up, +1 down
    new_idx = idx + direction
    if 0 <= idx < len(st.session_state.rules) and 0 <= new_idx < len(st.session_state.rules):
        st.session_state.rules[idx], st.session_state.rules[new_idx] = (
            st.session_state.rules[new_idx],
            st.session_state.rules[idx],
        )


def delete_rule(idx: int):
    if 0 <= idx < len(st.session_state.rules):
        del st.session_state.rules[idx]


def update_rule(idx: int, color: Tuple[int, int, int] = None, intensity: float = None, feather: int = None, enabled: bool = None, mask: np.ndarray = None, name: str = None):
    if 0 <= idx < len(st.session_state.rules):
        rule = st.session_state.rules[idx]
        if color is not None:
            rule.color = tuple(int(c) for c in color)
        if intensity is not None:
            rule.intensity = float(intensity)
        if feather is not None:
            rule.feather = int(feather)
        if enabled is not None:
            rule.enabled = bool(enabled)
        if mask is not None:
            rule.mask = mask.astype(np.float32)
        if name is not None:
            rule.name = name


def sidebar_controls():
    st.sidebar.header("Controls")
    st.sidebar.write("Selection Mode")
    st.session_state.selection_mode = st.sidebar.radio("Mode", ["Brush", "Magic Wand"], index=0, horizontal=True)

    if st.session_state.selection_mode == 'Magic Wand':
        st.sidebar.info("Click on the image to select a region. Use tolerance to adjust range.")
        tolerance = st.sidebar.slider("Tolerance", 0, 100, 15)
        return {'tolerance': tolerance}
    else:
        brush_radius = st.sidebar.slider("Brush radius (px)", 1, 80, 20)
        tool = st.sidebar.selectbox("Tool", ["Brush", "Rectangle", "Circle"], index=0)
        return {'brush_radius': brush_radius, 'tool': tool}

    st.sidebar.markdown("---")
    st.sidebar.subheader("Composition")
    mode = st.sidebar.radio("Mode", ["Priority (top overrides)", "Weighted Blend"], index=0)
    comp_mode = 'priority' if mode.startswith('Priority') else 'blend'
    st.session_state['composition_mode'] = comp_mode
    return {}


def show_rules_panel():
    st.sidebar.header("Color Rules")
    if st.sidebar.button("Clear all rules", type="secondary"):
        st.session_state.rules = []

    for idx, rule in enumerate(st.session_state.rules):
        with st.sidebar.expander(f"{rule.name}", expanded=False):
            col1, col2 = st.columns([1, 1])
            with col1:
                enabled = st.checkbox("Enabled", value=rule.enabled, key=f"enabled_{rule.id}")
            with col2:
                if st.button("Delete", key=f"delete_{rule.id}"):
                    delete_rule(idx)
                    st.experimental_rerun()
            name_val = st.text_input("Name", value=rule.name, key=f"name_{rule.id}")
            color = st.color_picker("Color", value="#%02x%02x%02x" % rule.color, key=f"color_{rule.id}")
            intensity = st.slider("Intensity", 0.0, 1.0, value=rule.intensity, key=f"intensity_{rule.id}")
            feather = st.slider("Feather (px)", 0, 100, value=int(rule.feather), key=f"feather_{rule.id}")
            colm1, colm2 = st.columns(2)
            with colm1:
                if st.button("Move Up", key=f"up_{rule.id}"):
                    reorder_rule(idx, -1)
                    st.experimental_rerun()
            with colm2:
                if st.button("Move Down", key=f"down_{rule.id}"):
                    reorder_rule(idx, 1)
                    st.experimental_rerun()
            update_rule(idx, color=tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), intensity=float(intensity), feather=int(feather), enabled=bool(enabled), name=name_val)

            st.markdown("---")
            st.caption("Refine mask with current selection")
            sel_mask = st.session_state.get('selection_mask', None)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                replace_dis = sel_mask is None
                if st.button("Replace", key=f"replace_{rule.id}", disabled=replace_dis):
                    if sel_mask is not None:
                        update_rule(idx, mask=sel_mask)
                        st.experimental_rerun()
            with c2:
                add_dis = sel_mask is None or rule.mask is None
                if st.button("Add", key=f"add_{rule.id}", disabled=sel_mask is None):
                    if sel_mask is not None:
                        base = rule.mask if rule.mask is not None else np.zeros_like(sel_mask)
                        new_mask = np.clip(base + sel_mask, 0.0, 1.0)
                        update_rule(idx, mask=new_mask)
                        st.experimental_rerun()
            with c3:
                if st.button("Subtract", key=f"sub_{rule.id}", disabled=sel_mask is None or rule.mask is None):
                    if sel_mask is not None and rule.mask is not None:
                        new_mask = np.clip(rule.mask - sel_mask, 0.0, 1.0)
                        update_rule(idx, mask=new_mask)
                        st.experimental_rerun()
            with c4:
                if st.button("Intersect", key=f"inter_{rule.id}", disabled=sel_mask is None or rule.mask is None):
                    if sel_mask is not None and rule.mask is not None:
                        new_mask = np.clip(rule.mask * sel_mask, 0.0, 1.0)
                        update_rule(idx, mask=new_mask)
                        st.experimental_rerun()

            if rule.mask is not None:
                st.caption("Mask preview")
                st.image((rule.mask * 255).astype(np.uint8), use_column_width=True, clamp=True)


def draw_brush_selection_area(image_rgb: np.ndarray, brush_radius: int, tool: str) -> np.ndarray:
    st.write("Draw on the image to create a selection mask.")
    # Use streamlit-drawable-canvas for brush drawing
    from streamlit_drawable_canvas import st_canvas

    h, w = image_rgb.shape[:2]
    pil_bg = Image.fromarray(image_rgb)
    drawing_mode = {
        'Brush': 'freedraw',
        'Rectangle': 'rect',
        'Circle': 'circle',
    }.get(tool, 'freedraw')
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.5)",
        stroke_width=int(brush_radius),
        stroke_color="#ffffff",
        background_color=None,  # keep transparent background to derive mask from alpha
        background_image=pil_bg,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    mask = None
    if canvas_result is not None and canvas_result.image_data is not None:
        rgba = canvas_result.image_data.astype(np.uint8)
        mask = make_mask_from_canvas_rgba(rgba)
    return mask


def click_magic_wand_area(image_rgb: np.ndarray, tolerance: int, gray: np.ndarray) -> np.ndarray:
    st.write("Click once to select region using magic wand.")
    # We can capture click via st.image and use returned coordinates
    # Streamlit returns a dict with x,y on click
    clicked = st.image(image_rgb, use_column_width=False, output_format="PNG")
    # Workaround: Streamlit doesn't provide native click coords from st.image.
    # We'll use a small input to accept coordinates.
    cols = st.columns([1, 1, 1])
    with cols[0]:
        x = st.number_input("x", min_value=0, max_value=image_rgb.shape[1]-1, value=int(image_rgb.shape[1]//2))
    with cols[1]:
        y = st.number_input("y", min_value=0, max_value=image_rgb.shape[0]-1, value=int(image_rgb.shape[0]//2))
    with cols[2]:
        st.write("Enter coordinates to apply magic wand.")
    if st.button("Apply Magic Wand"):
        return magic_wand_mask(gray, (int(x), int(y)), int(tolerance))
    return None


def main():
    init_state()
    st.title("Conditional Image Colorization (Luminance-Preserving)")

    with st.sidebar:
        st.subheader("Upload Grayscale Image")
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]) 
        if uploaded is not None:
            bgr = from_uploaded_file_to_bgr(uploaded.read())
            bgr = normalize_and_fit_image(bgr, max_width=1200, max_height=900)
            st.session_state.base_bgr = bgr
            st.session_state.work_bgr = bgr.copy()
            st.session_state.gray = ensure_gray(bgr)
        if st.button("Load sample grayscale"):
            # Generate a 256x256 radial gradient grayscale sample
            h, w = 256, 256
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2.0, w / 2.0
            dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            dist = (dist / dist.max())
            gray = (255 * (1.0 - dist)).astype(np.uint8)
            sample_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            sample_bgr = normalize_and_fit_image(sample_bgr, max_width=1200, max_height=900)
            st.session_state.base_bgr = sample_bgr
            st.session_state.work_bgr = sample_bgr.copy()
            st.session_state.gray = ensure_gray(sample_bgr)

    if st.session_state.base_bgr is None:
        st.info("Upload a grayscale (or color) image. The app will convert to grayscale for luminance.")
        st.stop()

    params = sidebar_controls()
    show_rules_panel()

    base_bgr = st.session_state.base_bgr
    base_rgb = to_display_rgb(base_bgr)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Selection")
        if st.session_state.selection_mode == 'Brush':
            mask = draw_brush_selection_area(base_rgb, brush_radius=params['brush_radius'], tool=params.get('tool', 'Brush'))
        else:
            mask = click_magic_wand_area(base_rgb, tolerance=params['tolerance'], gray=st.session_state.gray)

        st.session_state.selection_mask = mask
        with st.expander("Create color rule from selection", expanded=True):
            color_hex = st.color_picker("Pick color", value="#4aa3ff")
            intensity = st.slider("Intensity", 0.0, 1.0, 0.9)
            feather = st.slider("Feather (px)", 0, 100, 16)
            if st.button("Add Rule from Selection", type="primary", disabled=mask is None):
                rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                add_rule(mask, rgb, intensity, feather)

    with col_right:
        st.subheader("Preview & Export")
        # Compose rules to get colorized preview
        rules_dicts = [
            {
                'mask': r.mask,
                'color': r.color,
                'intensity': r.intensity,
                'feather': r.feather,
                'enabled': r.enabled,
            }
            for r in st.session_state.rules
        ]

        comp_mode = st.session_state.get('composition_mode', 'priority')
        colorized_bgr = compose_rules(st.session_state.gray, rules_dicts, mode=comp_mode)
        colorized_rgb = to_display_rgb(colorized_bgr)
        st.image(colorized_rgb, caption="Colorized preview", use_column_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reset Rules"):
                st.session_state.rules = []
                st.experimental_rerun()
        with col_b:
            file_bytes = save_image_bytes(colorized_bgr, ext='.png')
            st.download_button(
                label="Download PNG",
                data=file_bytes,
                file_name="colorized.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()



# Coral Detection on White Square Holders

This repository provides an automated image analysis pipeline for detecting and segmenting coral specimens held in white plastic square holders within aquarium rack photographs. Images captured at high resolution (7008 × 4672 px) are processed using two trained random-forest classifiers — one to identify the white holder frames, and one to segment the coral tissue within each frame — combined with morphological image operations to clean and sharpen the detected regions. The pipeline locates each holder in the image, assigns it a grid position label (R{row}C{col}), and extracts the precise polygon contour of the coral specimen inside it. These contours are exported as CSV files and loaded directly into ImageJ as named ROIs, where they can be overlaid on the original high-resolution images. The goal is to non-destructively measure coral shape and area across time points, enabling growth metrics such as surface area change and morphological development to be tracked without disturbing the specimens.

Note: a large neural network or transformer would provide higher quality detections, but would require GPU and more training. This tool is designed to be a relatively simple/low-tech/configurable approach that doesn't require a high-performing computer to rework. It's not a perfect tool, and will still require some "manual handling" of the images, but it's a quick and dirty solution that should help speed up the science.

Made by Dorian Tsai to support AIMS & SeaSim scientific experiments.


---

## Files

### Detection

| File | Purpose |
|---|---|
| `detect_corals_on_white_squares.py` | Main pipeline — run directly or called by the ImageJ macro |
| `run_detection.ijm` | ImageJ macro — calls the Python script and loads ROIs into ROI Manager |
| `debug_pipeline.py` | Saves step-by-step intermediate images to `debug/<image_stem>/` |

### Training

| File | Purpose |
|---|---|
| `label_annotate.py` | Samples training images and generates annotated label previews |
| `bbox_labeler.py` | Interactive GUI for drawing `white` / `coral` / `other` bounding boxes |
| `classifier_utils.py` | Shared feature extraction and training logic (imported by the two scripts below) |
| `train_white_classifier.py` | Trains the `WhiteTabClassifier` — white frame vs coral + other |
| `train_coral_classifier.py` | Trains the `CoralClassifier` — coral vs white frame + other |
| `test_classifier.py` | Applies a trained model to images and writes mask / overlay / compare outputs |

---

## Usage

### Detection

```bash
# Single image
python3 detect_corals_on_white_squares.py images/DSC00150.JPG output/

# Entire directory
python3 detect_corals_on_white_squares.py images/ output/

# Generate step-by-step debug images
python3 debug_pipeline.py images/ debug/
```

### Training workflow

```bash
# 1. Sample 10 images and create label templates (run once)
python3 label_annotate.py

# 2. Draw bounding boxes interactively
#    Keys: [w] white  [c] coral  [o] other
#          left-drag = draw  right-click = delete  [n]/[p] = next/prev  [q] = quit
python3 bbox_labeler.py

# 3. Train both classifiers
python3 train_white_classifier.py
python3 train_coral_classifier.py

# 4. Inspect masks on training images
python3 test_classifier.py
```

---

## Output per image

| File | Contents |
|---|---|
| `<stem>_annotated.jpg` | Original image with green ROI boxes, orange coral contours, and magenta calibration circles |
| `<stem>_rois.csv` | `row, col, x, y, width, height, center_x, center_y` (full-resolution pixels) |
| `<stem>_rois.json` | Same ROI data as JSON, consumed by the ImageJ macro |
| `<stem>_contours.csv` | Coral contour vertices: `row, col, point_idx, x, y` — one row per polygon point |
| `<stem>_calibration_rois.csv` | `side, idx, x, y, width, height, center_x, center_y, radius` — one row per calibration circle (8 total: L0–L3, R0–R3) |
| `<stem>_calibration_rois.json` | Same calibration data as JSON with image dimensions and full metadata |

### Using `_contours.csv` in ImageJ

```javascript
// Load the CSV, group rows by (row, col), then for each group:
makeSelection("polygon", xArray, yArray);
roiManager("add");
```

---

## Algorithm — Step by Step

All example images are from `DSC00150.JPG`.

---

### Step 1 — Downscale

The input image (7008 × 4672 px) is resized to **15%** of its original size
before any processing. All detection runs on this small version; coordinates
are scaled back to full resolution at the end. This gives ~45× speedup with
negligible accuracy loss for centimetre-scale objects.

![Step 1 — Downscaled image](readme/images/01_downscaled.jpg)

---

### Step 2 — White Pixel Mask (WhiteTabClassifier)

Every pixel in the downscaled image is classified as **white frame** (255) or
**other** (0) using a trained **random forest classifier** (`WhiteTabClassifier`).

Each pixel is described by nine features — B, G, R in BGR space; H, S, V in
HSV space; L, a, b in CIE L\*a\*b\* space — all normalised to \[0, 1\]. The forest
(200 trees, max depth 20) was trained on bounding-box labels drawn with
`bbox_labeler.py` using the `[w]hite` and `[o]ther` / `[c]oral` classes.

If no trained model is found, the pipeline falls back to an HSV brightness
threshold (Value ≥ 185, Saturation ≤ 65).

![Step 2 — White mask](readme/images/02_white_mask.jpg)

---

### Step 3a — Denoise (Median Blur)

A **5 × 5 median blur** removes salt-and-pepper noise introduced by the random
forest classifier without smearing the structural edges of the white frames.

![Step 3a — Denoised mask](readme/images/03a_denoised.jpg)

---

### Step 3b — Morphological Close

A **15 × 15 square closing** (dilation then erosion) seals remaining small
breaks in the white frame edges caused by lighting variation or reflections,
making each square frame a fully enclosed white region.

![Step 3b — After 15×15 close](readme/images/03b_morph_close.jpg)

---

### Step 3c — Inner Contour Detection (RETR_CCOMP Hierarchy)

`cv2.findContours` with `RETR_CCOMP` returns a two-level hierarchy:

- **Level 0 (outer)** — contours around white regions (the frames)
- **Level 1 (inner / child)** — contours around dark regions *enclosed* by white

Only level-1 contours are kept. These are the dark coral or reference dots
inside each white frame. Background darkness and open-sided regions are
automatically excluded. Many of these inner blobs are still noise — removed
in Steps 4 and 5.

![Step 3c — All inner contours](readme/images/03c_all_inner_contours.jpg)

---

### Step 4 — Size and Aspect-Ratio Filter

Each inner contour's bounding box is tested against two criteria:

| Criterion | Value | Reason |
|---|---|---|
| Area / image area | 0.2% – 10% | Rejects tiny noise and large merged regions |
| Aspect ratio (long / short side) | < 2.0 | Rejects long thin artefacts |

**Green = passes. Red = fails.**

![Step 4 — Size and aspect filter](readme/images/04_size_shape_filter.jpg)

---

### Step 5 — Four-Direction White Border Check

A thin strip is sampled on each of the **four sides** of every remaining blob.
Each strip must contain **≥ 8% white pixels**. The percentage shown is the
weakest of the four sides.

This removes dark regions open on one side — shadows against a frame edge, or
artefacts that pass the size filter but sit against a non-white background.

> **Note:** Calibration disc holders (grey circles on the side panels) also
> pass this check — they are enclosed by white on all sides. They are removed
> in Step 8.

![Step 5 — White border check](readme/images/05_white_border_check.jpg)

---

### Step 6 — Strategy Selection

The number of **distinct column positions** among surviving blobs is counted.
If **≥ 3 distinct columns** are found, the simple 15 × 15 close (Strategy 1)
is sufficient and the pipeline proceeds to Step 7.

If fewer than 3 columns are found — typically in close-up images where some
frame top edges are cropped — **Strategy 2 (directional close fallback)** tries
tall-narrow kernels (e.g. 20 × 4 px, then 4 × 20 px) in increasing sizes
(20, 30, 40, 50 px) and selects the one that produces the most columns.

![Step 6 — Strategy used](readme/images/06_strategy.jpg)

---

### Step 7 — 1D Clustering into Rows and Columns

The Y-centres of all blobs are clustered independently from the X-centres
using a gap threshold of 60% of the median blob dimension. The **median
position** of each cluster becomes a row or column centre.

Orange lines = row centres. Blue lines = column centres (before calibration
pruning).

![Step 7 — Clustering](readme/images/07_clustering.jpg)

---

### Step 8 — Calibration Column Pruning

Calibration disc holders sit *slightly closer* to the coral grid than coral
holders are to each other. The algorithm compares each edge column gap to the
**inner-spacing median** (coral-to-coral reference, excluding outermost gaps):

```
threshold = 0.93 × inner_median_spacing
```

Any outermost column whose gap is below this threshold is removed. Calibration
gaps have ratios ≤ 0.92; coral gaps ≥ 0.99 — a clean separation.

**Before pruning** — 6 columns detected:

![Step 8a — Before pruning](readme/images/08a_before_pruning.jpg)

**After pruning** — 2 red calibration columns removed, 4 green coral columns kept:

![Step 8b — After pruning](readme/images/08b_after_pruning.jpg)

---

### Step 9 — Grid Assignment

Each surviving blob is assigned to its nearest row and column centre, giving a
zero-based `(row, col)` index. Each grid cell is shown in a distinct colour.

![Step 9 — Grid assignment](readme/images/09_grid_assignment.jpg)

---

### Step 10 — Expand ROI to White Square Boundary

The raw blob bounding box covers only the **dark interior** (the coral). The
ROI is expanded outward by **15% of the blob dimensions** on each side to
include the white plastic frame walls:

```
border_x = median_blob_width  × 0.15
border_y = median_blob_height × 0.15
```

All coordinates are scaled back to full-resolution pixels. The result is a
tight ROI around each individual white square holder.

![Step 10 — Expanded ROIs](readme/images/10_expanded_rois.jpg)

---

### Step 11 — Coral Pixel Mask (CoralClassifier, raw)

Within each ROI's crop from the **15%-scale image**, the `CoralClassifier`
random forest labels every pixel as **coral** (255) or **other** (0). The same
nine-feature representation (B G R H S V L a\* b\*) is used as in Step 2.

The raw output typically contains salt-and-pepper noise and slightly ragged
edges — addressed in Step 12.

![Step 11 — Coral RF mask (raw)](readme/images/11_coral_raw_mask.jpg)

---

### Step 12 — Coral Mask Morphological Cleanup

Three operations round and clean the raw coral mask:

1. **Median blur (3 × 3)** — removes isolated noise pixels.
2. **Elliptical morphological close (7 × 7)** — seals small gaps and rounds
   the coral outline inward.
3. **Elliptical morphological open (3 × 3)** — removes small noise patches
   remaining after closing.

Elliptical structuring elements are used (rather than square) to produce a
smoother, more organic contour.

![Step 12 — Coral mask after cleanup](readme/images/12_coral_clean_mask.jpg)

---

### Step 13 — Coral Contour

The **largest connected dark region** in each cleaned coral mask is extracted
as a polygon contour, simplified to ~0.3% of the shorter ROI dimension, and
scaled back to full-resolution coordinates. The contour is drawn in orange on
the annotated output image and written to `_contours.csv`.

If no `CoralClassifier` model is found (i.e. before training), the pipeline
falls back to an **Otsu threshold** applied to the full-resolution ROI crop.

![Step 13 — Coral contours](readme/images/13_coral_contours.jpg)

---

### Step 14 — Calibration Circle Detection

Each image contains **8 calibration discs** — 4 stacked vertically on the left
side panel and 4 on the right — used to calibrate scene reflectivity. Discs
range from near-black through dark grey, medium grey, and near-white.

The search zone for each panel is a narrow horizontal strip: **4.0 – 10.6%
of image width from each edge**. The lower bound (4%) skips the corner bracket
structures at the image periphery; the upper bound (10.6%) stops before the
white plastic frame begins, which would otherwise produce strong false-positive
arc gradients.

Within each strip the y-range is limited to **8 – 58% of image height**,
excluding the top frame bolts and lower off-panel clutter.

Detection uses two complementary strategies applied to the cropped strip:

1. **Tophat + Blackhat morphology** — a combined disc-anomaly image highlights
   circular intensity deviations above *and* below the background level. This
   works for both bright (near-white) and dark discs against the grey panel.
2. **CLAHE on the raw strip** — used as a second source when the morphology
   signal is weak.

`HoughCircles` is run with progressively relaxed `param2` (30 → 22 → 16 → 12)
on both sources until ≥ 4 candidates are accumulated. A final
**best-4 selector** evaluates every combination of 4 from the candidate pool
using a composite score:

```
score = 1 / (1 + sp_cv + r_cv × 1.5 + x_spread × 0.5)
```

where `sp_cv` is the coefficient of variation of vertical spacings (penalises
uneven stacking), `r_cv` is radius consistency, and `x_spread` penalises
horizontal scatter.

Detected circles are written as **oval ROIs** (`makeOval`) to
`_calibration_rois.csv` / `_calibration_rois.json` and drawn in magenta on the
annotated image. The shaded zones in the debug image below show the left and
right search strips.

![Step 14 — Calibration circles](readme/images/14_calibration_circles.jpg)

---

### Final Output

Green boxes mark the white square ROI boundaries; orange outlines trace the
coral specimens; **magenta circles** mark the 8 calibration discs. Each ROI is
labelled with its `R{row}C{col}` grid position and each calibration circle with
its `{side}{idx}` label (L0–L3, R0–R3).

![Final annotated output](readme/images/final_annotated.jpg)

---

## Training the Classifiers

Both classifiers share the same nine-feature pixel representation and are
trained from bounding-box labels drawn over a sample of 10 images.

### Label classes

| Key | Label | Color | Used by |
|---|---|---|---|
| `w` | `white` | green | WhiteTabClassifier (positive), CoralClassifier (negative) |
| `c` | `coral` | orange | CoralClassifier (positive), WhiteTabClassifier (negative) |
| `o` | `other` | red | Both classifiers (negative) |

### Training data location

```
train/
  images/      10 randomly sampled full-resolution JPEGs
  labels/      one JSON per image with labeled ROI bounding boxes
  models/      YYYYMMDD_WhiteTabClassifier.joblib
               YYYYMMDD_CoralClassifier.joblib
  annotated/   label preview images (generated by label_annotate.py)
  output_masks/ mask / overlay / compare images (generated by test_classifier.py)
```

### JSON label format

```json
{
  "image": "train/images/DSC00150.JPG",
  "image_width": 7008,
  "image_height": 4672,
  "rois": [
    {"label": "white", "row": 0, "col": 0, "x": 1666, "y": 853,  "width": 1013, "height": 773},
    {"label": "coral",                      "x": 1750, "y": 950,  "width":  800, "height":  600},
    {"label": "other",                      "x":  100, "y": 100,  "width":  400, "height":  400}
  ]
}
```

---

## Known Limitation

Squares whose interior is **empty** (fully white) or contains only a very small
dark reference dot are not detected. The hole-in-white approach requires a dark
occupant large enough to form a detectable enclosed blob.

---

## Running via ImageJ (`run_detection.ijm`)

The macro calls `detect_corals_on_white_squares.py` from inside ImageJ, then
automatically loads the resulting ROIs into the **ROI Manager** as named regions.

---

### How it works

The macro locates `detect_corals_on_white_squares.py` in ImageJ's own `macros/`
folder, then calls it via ImageJ's `exec()` function, passing the image (or
directory) path and the output folder as separate arguments. When the script
finishes, the macro reads `<stem>_rois.csv` and `<stem>_contours.csv` and
populates the ROI Manager with:

- `R{row}C{col}` — rectangle around each white square holder
- `coral_R{row}C{col}` — polygon tracing the coral specimen inside it

---

### Required folder layout inside `macros/`

Both the macro and the Python script must live in ImageJ's `macros/` directory.
If you have trained classifiers, the model files go in a `train/models/`
subdirectory **relative to the script**:

```
macros/
├── run_detection.ijm
├── detect_corals_on_white_squares.py
└── train/
    └── models/
        ├── YYYYMMDD_WhiteTabClassifier.joblib
        └── YYYYMMDD_CoralClassifier.joblib
```

Without the model files the pipeline falls back to HSV thresholding (white mask)
and Otsu thresholding (coral contour), which still produces usable results.

---

### Platform setup

#### Ubuntu (Linux)

**1. Install Python dependencies**

```bash
pip3 install opencv-python numpy scikit-learn joblib
```

If you use a conda environment, activate it first:

```bash
conda activate <your-env>
conda install -c conda-forge opencv numpy scikit-learn joblib
```

**2. Find the ImageJ macros folder**

The default locations depend on how ImageJ was installed:

| Installation | Macros folder |
|---|---|
| Standalone ImageJ `.tar.gz` | `~/ImageJ/macros/` |
| Fiji (recommended) | `~/Fiji.app/macros/` |
| System package | `/usr/share/imagej/macros/` |

To confirm, open ImageJ and check the title bar — it shows the ImageJ root
directory. The `macros/` folder is directly inside it.

**3. Copy the files**

```bash
# Standalone ImageJ example — adjust path if using Fiji
cp detect_corals_on_white_squares.py ~/ImageJ/macros/
cp run_detection.ijm                 ~/ImageJ/macros/

# Copy trained models if available
mkdir -p ~/ImageJ/macros/train/models/
cp train/models/*.joblib ~/ImageJ/macros/train/models/
```

**4. Run the macro**

Open ImageJ, then: **Plugins → Macros → Run…** → select `run_detection.ijm`.

In the dialog:
- **Process**: choose `Current image` (must have an image open) or
  `All images in a directory`.
- **Output folder**: `Same folder as image` writes outputs alongside the input;
  `Choose folder…` opens a folder picker.
- **Python command**: leave as `python3` for the system Python. If you use a
  conda environment, enter the full path, e.g.
  `~/anaconda3/envs/myenv/bin/python3`.

---

#### macOS

**1. Install Python dependencies**

With Homebrew Python:

```bash
pip3 install opencv-python numpy scikit-learn joblib
```

With conda:

```bash
conda activate <your-env>
conda install -c conda-forge opencv numpy scikit-learn joblib
```

> **macOS note**: Apple's built-in `python3` (from Command Line Tools) works but
> may not have `cv2`. Installing via Homebrew (`brew install python`) or using a
> conda environment is recommended.

**2. Find the ImageJ macros folder**

| Installation | Macros folder |
|---|---|
| Fiji (recommended) | `/Applications/Fiji.app/macros/` |
| Standalone ImageJ `.app` | `/Applications/ImageJ.app/macros/` |

**3. Copy the files**

```bash
# Fiji example
cp detect_corals_on_white_squares.py /Applications/Fiji.app/macros/
cp run_detection.ijm                 /Applications/Fiji.app/macros/

mkdir -p /Applications/Fiji.app/macros/train/models/
cp train/models/*.joblib /Applications/Fiji.app/macros/train/models/
```

**4. Run the macro**

Open Fiji/ImageJ, then: **Plugins → Macros → Run…** → select `run_detection.ijm`.

In the dialog, the **Python command** field may need the full path if ImageJ
cannot resolve `python3` from its environment:

```
/usr/local/bin/python3          # Homebrew on Intel Mac
/opt/homebrew/bin/python3       # Homebrew on Apple Silicon
/Users/<you>/anaconda3/envs/<env>/bin/python3   # conda environment
```

To find the correct path, open a Terminal and run `which python3`.

---

#### Windows 11

**1. Install Python**

Download Python 3.x from [python.org](https://www.python.org/downloads/). During
installation, tick **"Add Python to PATH"**.

Then install the required packages from a Command Prompt or PowerShell:

```cmd
pip install opencv-python numpy scikit-learn joblib
```

With Anaconda/Miniconda, open an Anaconda Prompt:

```cmd
conda activate <your-env>
conda install -c conda-forge opencv numpy scikit-learn joblib
```

**2. Find the ImageJ macros folder**

| Installation | Macros folder |
|---|---|
| Fiji (recommended) | `C:\Users\<you>\Fiji.app\macros\` |
| Standalone ImageJ | `C:\Program Files\ImageJ\macros\` |

> If ImageJ was installed to `Program Files`, you may need to run it as
> Administrator to write files there, or install to a user-writable location
> such as `C:\Users\<you>\ImageJ\`.

**3. Copy the files**

Using File Explorer: copy `run_detection.ijm` and
`detect_corals_on_white_squares.py` into the `macros\` folder found above.

If you have trained models, create `macros\train\models\` and copy the
`.joblib` files there.

**4. Run the macro**

Open Fiji/ImageJ, then: **Plugins → Macros → Run…** → select `run_detection.ijm`.

In the dialog, set **Python command** to whichever of the following resolves in
your environment:

| Scenario | Python command to enter |
|---|---|
| Python added to PATH | `python` |
| Python **not** in PATH | Full path, e.g. `C:\Python312\python.exe` |
| Anaconda (base env) | `C:\Users\<you>\anaconda3\python.exe` |
| Anaconda (named env) | `C:\Users\<you>\anaconda3\envs\<env>\python.exe` |

To confirm the correct path, open a Command Prompt and run `where python`.

> **Windows note**: ImageJ's `exec()` launches the Python process directly
> (without the Command Prompt window) using Java's `Runtime.exec()`. This means
> the PATH seen by Python is inherited from ImageJ's process, which may differ
> from your shell PATH. When in doubt, use the full path to `python.exe`.

---

### Dialog reference

| Field | Options | Notes |
|---|---|---|
| **Process** | `Current image` / `All images in a directory` | Current image requires an open image |
| **Output folder** | `Same folder as image` / `Choose folder…` | Choose folder opens a system folder picker |
| **Python command** | any string | Path to the Python executable; see platform notes above |

---

### Troubleshooting

**No output files appear after "Detection complete."**

- Verify the Python command resolves correctly for your platform (see above).
- Run the equivalent command in a terminal to see any error output:
  ```bash
  python3 /path/to/macros/detect_corals_on_white_squares.py /path/to/image.jpg /path/to/output/
  ```
- Confirm `detect_corals_on_white_squares.py` is in the same `macros/` folder
  as `run_detection.ijm` — the macro looks for it there by design.

**ROI Manager is empty after detection**

The macro only loads ROIs in `Current image` mode. In directory mode, outputs are
written to disk but not loaded into ImageJ.

**`cv2` / `numpy` not found**

The Python installation that ImageJ calls may differ from the one you tested in
the terminal. Use the full absolute path to the Python executable in the dialog
(e.g. `/home/user/anaconda3/envs/coral/bin/python3`).

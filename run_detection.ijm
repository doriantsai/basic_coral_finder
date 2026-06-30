// run_detection.ijm
// ImageJ macro: detects white square coral holders and coral specimens in the
// current image (or a chosen directory) by calling the external Python
// detection script, then loads the resulting ROIs into ImageJ's ROI Manager.
//
// Requirements:
//   - Python 3 with opencv-python, numpy, scikit-learn, joblib installed
//   - detect_corals_on_white_squares.py in the same folder as this macro
//   - train/models/  containing the trained WhiteTabClassifier and
//     CoralClassifier .joblib files, relative to the macro folder
//
// ROI Manager after loading (single-image mode):
//   R{r}C{c}        — rectangle covering the white square holder
//   coral_R{r}C{c}  — polygon tracing the coral specimen inside that holder

// ── Locate this macro's directory to find the Python script ──────────────────
macro_dir   = getDirectory("macros");
script_path = macro_dir + "detect_corals_on_white_squares.py";

// ── Dialog ────────────────────────────────────────────────────────────────────
Dialog.create("Coral on White Square Detector");
Dialog.addChoice("Process", newArray("Current image", "All images in a directory"),
                 "Current image");
Dialog.addChoice("Output folder", newArray("Same folder as image", "Choose folder..."),
                 "Same folder as image");
Dialog.addString("Python command", "python3", 10);
Dialog.show();

mode        = Dialog.getChoice();
out_choice  = Dialog.getChoice();
python_cmd  = Dialog.getString();

// ── Resolve input / output paths ──────────────────────────────────────────────
if (mode == "Current image") {
    if (nImages == 0)
        exit("No image is open.  Open an image first.");

    img_dir = getDirectory("image");   // directory with trailing separator; "" if unsaved
    if (img_dir == "") {
        // Image not saved to disk — write a temporary copy
        img_path = getDirectory("temp") + "ij_temp_input.tif";
        saveAs("Tiff", img_path);
        img_dir  = getDirectory("temp");
    } else {
        img_path = img_dir + getTitle();
    }

    if (out_choice == "Choose folder...")
        output_dir = getDirectory("Select output folder");
    else
        output_dir = img_dir;
} else {
    input_dir  = getDirectory("Select input image directory");
    if (out_choice == "Choose folder...")
        output_dir = getDirectory("Select output folder");
    else
        output_dir = input_dir;
    img_path = input_dir;   // used only for stem derivation below; overridden in dir mode
}

// ── Run Python detection script ───────────────────────────────────────────────
// exec() with separate arguments uses Runtime.exec(String[]) — each path is
// passed directly to Python without shell tokenisation, so spaces in paths and
// quote characters are never an issue.
print("Running: " + python_cmd + " " + script_path + " " + img_path + " " + output_dir);
exec(python_cmd, script_path, img_path, output_dir);
print("Detection complete.");

// ── Load ROIs into ROI Manager (single-image mode only) ──────────────────────
if (mode == "Current image") {
    stem     = File.getNameWithoutExtension(img_path);
    csv_path = output_dir + stem + "_rois.csv";

    if (!File.exists(csv_path)) {
        showMessage("Detection finished",
                    "ROI file not found:\n" + csv_path +
                    "\n\nCheck the Log window for errors.");
    } else {
        roiManager("reset");

        // ── White square bounding boxes (_rois.csv) ───────────────────────────
        // Columns: row, col, x, y, width, height, center_x, center_y, ...
        lines = split(File.openAsString(csv_path), "\n");
        n_rois = 0;
        for (i = 1; i < lines.length; i++) {
            line = trim(lines[i]);
            if (line == "") continue;
            f = split(line, ",");
            if (f.length < 6) continue;
            row_id = parseInt(f[0]);
            col_id = parseInt(f[1]);
            rx     = parseInt(f[2]);
            ry     = parseInt(f[3]);
            rw     = parseInt(f[4]);
            rh     = parseInt(f[5]);
            makeRectangle(rx, ry, rw, rh);
            roiManager("add");
            n = roiManager("count");
            roiManager("select", n - 1);
            roiManager("rename", "R" + row_id + "C" + col_id);
            n_rois++;
        }
        print("Loaded " + n_rois + " white square ROI(s).");

        // ── Coral contour polygons (_contours.csv) ────────────────────────────
        // Columns: row, col, point_idx, x, y  (one vertex per line)
        // Rows are ordered by (row, col) then point_idx; consecutive rows with
        // the same (row, col) form one polygon.
        cont_path = output_dir + stem + "_contours.csv";
        if (File.exists(cont_path)) {
            clines  = split(File.openAsString(cont_path), "\n");
            cur_row = -1;  cur_col = -1;
            xs = newArray(0);  ys = newArray(0);
            n_contours = 0;

            for (i = 1; i < clines.length; i++) {
                line = trim(clines[i]);
                if (line == "") continue;
                f = split(line, ",");
                if (f.length < 5) continue;
                r  = parseInt(f[0]);  c  = parseInt(f[1]);
                px = parseInt(f[3]);  py = parseInt(f[4]);

                // When (row, col) changes, flush the accumulated polygon
                if (r != cur_row || c != cur_col) {
                    if (xs.length > 2) {
                        makeSelection("polygon", xs, ys);
                        roiManager("add");
                        n = roiManager("count");
                        roiManager("select", n - 1);
                        roiManager("rename", "coral_R" + cur_row + "C" + cur_col);
                        n_contours++;
                    }
                    cur_row = r;  cur_col = c;
                    xs = newArray(0);  ys = newArray(0);
                }
                xs = Array.concat(xs, px);
                ys = Array.concat(ys, py);
            }
            // Flush the final polygon
            if (xs.length > 2) {
                makeSelection("polygon", xs, ys);
                roiManager("add");
                n = roiManager("count");
                roiManager("select", n - 1);
                roiManager("rename", "coral_R" + cur_row + "C" + cur_col);
                n_contours++;
            }
            print("Loaded " + n_contours + " coral contour(s).");
        } else {
            print("No contours file found: " + cont_path);
        }

        // ── Calibration circle ovals (_calibration_rois.csv) ─────────────────
        // Columns: side, idx, x, y, width, height, center_x, center_y, radius, ...
        // x,y is the top-left of the bounding square; width==height==2*radius.
        cal_path = output_dir + stem + "_calibration_rois.csv";
        if (File.exists(cal_path)) {
            cal_lines = split(File.openAsString(cal_path), "\n");
            n_cal = 0;
            for (i = 1; i < cal_lines.length; i++) {
                line = trim(cal_lines[i]);
                if (line == "") continue;
                f = split(line, ",");
                if (f.length < 9) continue;
                side  = f[0];
                idx   = f[1];
                ox    = parseInt(f[2]);
                oy    = parseInt(f[3]);
                ow    = parseInt(f[4]);
                oh    = parseInt(f[5]);
                makeOval(ox, oy, ow, oh);
                roiManager("add");
                n = roiManager("count");
                roiManager("select", n - 1);
                roiManager("rename", "cal_" + side + idx);
                n_cal++;
            }
            print("Loaded " + n_cal + " calibration circle ROI(s).");
        } else {
            print("No calibration ROI file found: " + cal_path);
        }

        roiManager("show all");

        // Open the annotated image alongside the original
        ann_path = output_dir + stem + "_annotated.jpg";
        if (File.exists(ann_path))
            open(ann_path);
    }
}

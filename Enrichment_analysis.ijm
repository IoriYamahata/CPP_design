
CPP_CH      = 2;   // 1=C1, 2=C2, 3=C3
RECEPTOR_CH = 3;   // 1=C1, 2=C2, 3=C3

ROLLING_BALL = 50;     // pixels
WHOLE_MIN_SIZE = 20;   // pixels^2
RECEPTOR_MIN_SIZE = 0.1; // pixels^2
RECEPTOR_CIRC_MIN = 0.00;
RECEPTOR_CIRC_MAX = 1.00;

DEBUG_SAVE_MASKS = true;
DEBUG_PRINT_TITLES = true;

WHOLE_THRESH  = "Huang dark";
RECEPT_THRESH = "Otsu dark";

SAVE_SEPARATE_OVERLAYS = true;

function selectRoisByIndexArray(idxArr) {
    roiManager("Deselect");
    roiManager("Select", idxArr);
}

function stripExt(fname) {
    dot = lastIndexOf(fname, ".");
    if (dot == -1) return fname;
    return substring(fname, 0, dot);
}

function openAny(fullpath) {
    lower = toLowerCase(fullpath);
    if (endsWith(lower, ".oir")) {
        // ほぼデフォルト相当（UI止まり回避で view と stack_order だけ固定）
        run("Bio-Formats Importer",
            "open=[" + fullpath + "] view=Hyperstack stack_order=XYCZT");
    } else {
        open(fullpath);
    }
}

// 単一チャンネル画像からタイルを作る（複製→矩形→Crop）
function makeTileFromSingleChannel(srcTitle, outTitle, x0, y0, w, h) {
    selectWindow(srcTitle);
    run("Duplicate...", "title=" + outTitle);
    selectWindow(outTitle);
    makeRectangle(x0, y0, w, h);
    run("Crop");
    return outTitle;
}

// --------------------
// Main
// --------------------
requires("1.53");
setBatchMode(true);
run("Set Measurements...", "mean area decimal=6");

inputDir = getDirectory("Choose folder containing tif/tiff/oir files");
if (inputDir=="") exit("No folder selected.");

outDir = getDirectory("Choose folder to save CSV/PNGs");
if (outDir=="") exit("No output folder selected.");

outPath = outDir + "enrichment_results_tiles.csv";
File.saveString("filename,tile,x0,y0,I_Total,I_Receptor,Ratio,WholeCellArea,ReceptorArea,Flags\n", outPath);

list = getFileList(inputDir);

for (i=0; i<list.length; i++) {

    name = list[i];
    lower = toLowerCase(name);

    is_tif = (endsWith(lower, ".tif") || endsWith(lower, ".tiff"));
    is_oir = endsWith(lower, ".oir");
    if (!(is_tif || is_oir)) continue;

    fullpath = inputDir + name;

    run("Close All");
    roiManager("Reset");
    run("Clear Results");

    openAny(fullpath);
    if (nImages==0) {
        File.append(name + ",NA,NA,NA,NA,NA,NA,NA,NA,OPEN_FAILED;\n", outPath);
        continue;
    }

    baseTitle = getTitle();
    W = getWidth();
    H = getHeight();

    if ((W % 2)!=0 || (H % 2)!=0) {
        File.append(name + ",NA,NA,NA,NA,NA,NA,NA,NA,ODD_DIM;\n", outPath);
        run("Close All");
        continue;
    }

    tileW = W/2;
    tileH = H/2;

    // まず全体をSplit（ここが「1回だけ」）
    run("Split Channels");
    // Split後のチャンネル窓タイトルを素直に拾う（C1- 形式を優先）
    titles = getList("image.titles");
    c1full = ""; c2full = ""; c3full = "";
    for (tt=0; tt<titles.length; tt++) {
        if (startsWith(titles[tt], "C1-")) c1full = titles[tt];
        if (startsWith(titles[tt], "C2-")) c2full = titles[tt];
        if (startsWith(titles[tt], "C3-")) c3full = titles[tt];
    }

    if (DEBUG_PRINT_TITLES) {
        print("FILE=" + name);
        print(" baseTitle=" + baseTitle);
        print(" c1full=" + c1full);
        print(" c2full=" + c2full);
        print(" c3full=" + c3full);
    }

    if (c1full=="" || c2full=="" || c3full=="") {
        File.append(name + ",NA,NA,NA,NA,NA,NA,NA,NA,SKIP_NOT_3CH;\n", outPath);
        run("Close All");
        continue;
    }

    if (RECEPTOR_CH==1) recFull = c1full;
    else if (RECEPTOR_CH==2) recFull = c2full;
    else recFull = c3full;

    if (CPP_CH==1) cppFull = c1full;
    else if (CPP_CH==2) cppFull = c2full;
    else cppFull = c3full;

    tileLabels = newArray("TL","TR","BL","BR");
    tileX = newArray(0, tileW, 0, tileW);
    tileY = newArray(0, 0, tileH, tileH);

    for (t=0; t<4; t++) {

        roiManager("Reset");
        run("Clear Results");
        flags = "";

        tileLab = tileLabels[t];
        x0 = tileX[t];
        y0 = tileY[t];

        // ここで「各チャンネル単体」からタイルを作る（Splitしない）
        recTile = makeTileFromSingleChannel(recFull, "REC_TILE_" + tileLab, x0, y0, tileW, tileH);
        cppTile = makeTileFromSingleChannel(cppFull, "CPP_TILE_" + tileLab, x0, y0, tileW, tileH);

        // -------------------------
        // Phase 1: Background correction
        // -------------------------
        selectWindow(recTile);
        run("Subtract Background...", "rolling=" + ROLLING_BALL);

        selectWindow(cppTile);
        run("Subtract Background...", "rolling=" + ROLLING_BALL);

        // measurement用に複製
        selectWindow(cppTile);
        run("Duplicate...", "title=CPP_meas");

        // -------------------------
        // Phase 2: Whole cell mask from CPP
        // -------------------------
        selectWindow(cppTile);
        run("Duplicate...", "title=CPP_for_mask");
        selectWindow("CPP_for_mask");

        setAutoThreshold(WHOLE_THRESH);
        run("Convert to Mask");

        if (DEBUG_SAVE_MASKS) {
            saveAs("PNG", outDir + stripExt(name) + "_" + tileLab + "_CPP_mask.png");
        }

        run("Analyze Particles...", "size=" + WHOLE_MIN_SIZE + "-Infinity show=Nothing add");

        wholeCount = roiManager("count");
        hasWhole = (wholeCount > 0);

        if (!hasWhole) {
            flags = flags + "NO_WHOLE_CELL_ROI;";
        } else {
            idx = newArray(wholeCount);
            for (k=0; k<wholeCount; k++) idx[k] = k;
            selectRoisByIndexArray(idx);
            roiManager("Combine");
            roiManager("Add");
            keep = roiManager("count") - 1;

            for (k=keep-1; k>=0; k--) {
                roiManager("Select", k);
                roiManager("Delete");
            }
            // Whole ROI は index 0
        }

        // -------------------------
        // Phase 3: Receptor mask from receptor channel
        // -------------------------
        selectWindow(recTile);
        run("Duplicate...", "title=REC_for_mask");
        selectWindow("REC_for_mask");

        setAutoThreshold(RECEPT_THRESH);
        run("Convert to Mask");

        if (DEBUG_SAVE_MASKS) {
            saveAs("PNG", outDir + stripExt(name) + "_" + tileLab + "_REC_mask.png");
        }

        run("Analyze Particles...", "size=" + RECEPTOR_MIN_SIZE + "-Infinity circularity=" + RECEPTOR_CIRC_MIN + "-" + RECEPTOR_CIRC_MAX + " show=Nothing add");

        totalRois = roiManager("count");
        if (hasWhole) recStart = 1;
        else recStart = 0;

        recCount = totalRois - recStart;
        hasRec = (recCount > 0);

        if (!hasRec) {
            flags = flags + "NO_RECEPTOR_ROI;";
        } else {
            idx2 = newArray(recCount);
            for (k=0; k<recCount; k++) idx2[k] = recStart + k;
            selectRoisByIndexArray(idx2);
            roiManager("Combine");
            roiManager("Add");
            keep2 = roiManager("count") - 1;

            for (k=keep2-1; k>=recStart; k--) {
                roiManager("Select", k);
                roiManager("Delete");
            }
            // hasWhole なら receptor ROI は index 1、なければ index 0
        }

        // -------------------------
        // Measure on CPP_meas
        // -------------------------
        I_Total = "NA"; I_Rec = "NA"; ratio = "NA";
        wholeArea = "NA"; recArea = "NA";

        selectWindow("CPP_meas");

        if (hasWhole) {
            roiManager("Select", 0);
            run("Measure");
            row = nResults - 1;
            I_Total = getResult("Mean", row);
            wholeArea = getResult("Area", row);
            if (I_Total <= 0) flags = flags + "ZERO_TOTAL;";
        }

        if (hasRec) {
            if (hasWhole) recIndex = 1;
            else recIndex = 0;

            roiManager("Select", recIndex);
            run("Measure");
            row = nResults - 1;
            I_Rec = getResult("Mean", row);
            recArea = getResult("Area", row);
        }

        if (I_Total!="NA" && I_Rec!="NA") {
            if (I_Total > 0) ratio = I_Rec / I_Total;
            else ratio = "NA";
        }

        // -------------------------
        // Save overlays (optional)
        // -------------------------
        if (SAVE_SEPARATE_OVERLAYS) {

            if (hasWhole) {
                selectWindow("CPP_meas");
                run("Duplicate...", "title=CPP_OVERLAY_TMP");
                selectWindow("CPP_OVERLAY_TMP");
                run("RGB Color");
                roiManager("Select", 0);
                setColor(255, 255, 0);
                run("Draw", "slice");
                saveAs("PNG", outDir + stripExt(name) + "_" + tileLab + "_CPP_wholeROI.png");
                close();
            }

            if (hasRec) {
                recIndex = 0;
                if (hasWhole) recIndex = 1;

                selectWindow(recTile);
                run("Duplicate...", "title=REC_OVERLAY_TMP");
                selectWindow("REC_OVERLAY_TMP");
                run("RGB Color");
                roiManager("Select", recIndex);
                setColor(255, 255, 0);
                run("Draw", "slice");
                saveAs("PNG", outDir + stripExt(name) + "_" + tileLab + "_REC_receptorROI.png");
                close();
            }
        }

        // write CSV row
        line = name + "," + tileLab + "," + x0 + "," + y0 + "," +
               I_Total + "," + I_Rec + "," + ratio + "," + wholeArea + "," + recArea + "," + flags + "\n";
        File.append(line, outPath);

        // タイル関連だけ閉じる（元のSplitチャンネルは残す）
        if (isOpen("CPP_meas")) close("CPP_meas");
        if (isOpen("CPP_for_mask")) close("CPP_for_mask");
        if (isOpen("REC_for_mask")) close("REC_for_mask");
        if (isOpen(recTile)) close(recTile);
        if (isOpen(cppTile)) close(cppTile);
    }

    run("Close All");
}

setBatchMode(false);
print("Done. Results saved to: " + outPath);
run("Subtract Background...", "rolling=50 stack");
run("Remove Outliers...", "radius=2 threshold=10 which=Bright stack");
run("OME-TIFF...", "save=[C:/Users/charl/Documents/Current_work/DeepLabCut_etc/DeepLabCut-fork/examples/WBFM_fromCBE/img - denoised.tif] export compression=Uncompressed");
run("FeatureJ Hessian", "largest smoothing=1.0");
run("OME-TIFF...", "save=[C:/Users/charl/Documents/Current_work/DeepLabCut_etc/DeepLabCut-fork/examples/WBFM_fromCBE/img - denoised - Hessian.tif] export compression=Uncompressed");
run("Extended Min & Max 3D", "operation=[Extended Minima] dynamic=10 connectivity=6");
run("OME-TIFF...", "save=[C:/Users/charl/Documents/Current_work/DeepLabCut_etc/DeepLabCut-fork/examples/WBFM_fromCBE/img - denoised - Hessian - emin.tif] export compression=Uncompressed");
run("3D Objects Counter", "threshold=128 slice=19 min.=1 max.=21547500 exclude_objects_on_edges centroids statistics summary");
saveAs("Results", "C:/Users/charl/Documents/Current_work/DeepLabCut_etc/DeepLabCut-fork/examples/WBFM_fromCBE/Statistics for img.csv");

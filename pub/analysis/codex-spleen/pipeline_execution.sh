#!/usr/bin/env bash
# bash -e /lab/repos/cytokit/pub/analysis/codex-spleen/pipeline_execution.sh

# Ignore warnings relating to how the CODEX tif files were originally saved
export PYTHONWARNINGS='ignore::FutureWarning:h5py,ignore:ImageJ tags do not contain "axes" property:UserWarning:__main__'

for EXPERIMENT in "20180101_codex_mouse_spleen_balbc_slide1"
do
    DATA_DIR=$CYTOKIT_DATA_DIR/20180101_codex_spleen/$EXPERIMENT
    BASE_CONF=$CYTOKIT_REPO_DIR/pub/config/codex-spleen/experiment.yaml
    
    # Generate configurations for experiment variants
    # v00: Process data as-is with CellProfiler quantification (and DB export for CPA)
    # v01: Run with drift compensation and deconvolution (primarily for performance benchmarking)
    cytokit config editor --base-config-path=$BASE_CONF --output-dir=$DATA_DIR/output \
    add analysis '{"cellprofiler_quantification": {"export_db": True, "export_csv": True, "export_db_objects_separately": True}}' \
    save_variant v00/config reset \
    set processor.args.run_drift_comp True \
    set processor.args.run_deconvolution True \
    save_variant v01/config reset \
    exit
    
    # Run processing for each variant of this experiment
    for VARIANT in v00 v01
    do
        OUTPUT_DIR=$DATA_DIR/output/$VARIANT
        CONFIG_DIR=$OUTPUT_DIR/config
        echo "Processing experiment $EXPERIMENT (variant = $VARIANT, config = $CONFIG_DIR)"
        
        # Symlink to downloaded tile images; Typically tiles would not already have been
        # assembled from individual channel images but as this was already done in the data
        # shared for the CODEX publication, the resulting images can be used as-is
        echo "Creating symlinks from raw data to $OUTPUT_DIR/processor/tile"
        mkdir -p $OUTPUT_DIR/processor/tile
        for f in `ls $DATA_DIR/raw/*.tif`; do
            FILENAME=`basename $f | sed 's/BALBc-1/R01/g'`
            LINK_PATH=$OUTPUT_DIR/processor/tile/$FILENAME
            if [ ! -e "$LINK_PATH" ]; then
                ln -s $f $LINK_PATH
            fi
        done
        
        # Note here that the data dir for the processor command is equal to output dir
        echo "Running analysis"
        cytokit processor run_all --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR --output-dir=$OUTPUT_DIR
        cytokit operator run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
        cytokit analysis run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
    done
done

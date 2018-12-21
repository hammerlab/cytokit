#!/usr/bin/env bash
# bash -e /lab/repos/cell-image-analysis/analysis/experiments/20180101_codex_mouse_spleen/pipeline_execution.sh

# First line in logs: 2018-10-24 00:46:21,852:INFO:25690:

# Experiments:
# 20180101_codex_mouse_spleen_balbc_slide1

for EXPERIMENT in "20180101_codex_mouse_spleen_balbc_slide1"
do
    DATA_DIR=$CYTOKIT_DATA_DIR/20180101_codex_spleen/$EXPERIMENT
    BASE_CONF=$CYTOKIT_ANALYSIS_REPO_DIR/config/experiment/20180101_codex_spleen/$EXPERIMENT/experiment.yaml
    
    # Generate configurations for experiment variants
    cytokit config editor --base-config-path=$BASE_CONF --output-dir=$DATA_DIR/output \
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
        
        echo "Creating symlinks from raw data to $OUTPUT_DIR/processor/tile"
        for f in `ls $DATA_DIR/raw/*.tif`; do
            LINK_PATH=$OUTPUT_DIR/processor/tile/$(basename $f)
            if [ ! -e "$LINK_PATH" ]; then
                ln -s $f $LINK_PATH
            fi
        done
        
        # Note here that the data dir for the processor command is equal to output dir
        echo "Running analysis"
        cytokit processor run_all --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR --output-dir=$OUTPUT_DIR
        #cytokit operator run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
        #cytokit analysis run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
    done
done

#!/usr/bin/env bash
# bash -e /lab/repos/cytokit/pub/analysis/cell-profiler/pipeline_execution.sh

for EXPERIMENT in "20180101_codex_mouse_spleen_balbc_slide1"
do
    DATA_DIR=$CYTOKIT_DATA_DIR/20180101_codex_spleen/$EXPERIMENT
    BASE_CONF=$CYTOKIT_REPO_DIR/pub/config/codex-spleen/experiment.yaml
    
    # Generate configurations with cytometer implementation 
    cytokit config editor --base-config-path=$BASE_CONF --output-dir=$DATA_DIR/output \
    set processor.cytometry.type '{"module": "cp_cytometer", "class": "CPCytometer"}' \
    set processor.gpus [0] \
    save_variant v02/config reset \
    exit
    
    # Add this dir to python path to pick up custom cytometer implementation
    export PYTHONPATH=$CYTOKIT_REPO_DIR/pub/analysis/cell-profiler
    OUTPUT_DIR=$DATA_DIR/output/v02
    CONFIG_DIR=$OUTPUT_DIR/config

    # Symlink to downloaded tile images
    echo "Creating symlinks from raw data to $OUTPUT_DIR/processor/tile"
    mkdir -p $OUTPUT_DIR/processor/tile
    for f in `ls $DATA_DIR/raw/*.tif`; do
        LINK_PATH=$OUTPUT_DIR/processor/tile/$(basename $f)
        if [ ! -e "$LINK_PATH" ]; then
            ln -s $f $LINK_PATH
        fi
    done

    # Note here that the data dir for the processor command is equal to output dir
    echo "Running analysis"
    cytokit processor run_all --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR --output-dir=$OUTPUT_DIR --tile-indexes=[1]
done

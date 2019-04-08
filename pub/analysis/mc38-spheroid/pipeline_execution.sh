#!/usr/bin/env bash
# cd $CYTOKIT_REPO_DIR/pub/analysis/mc38-spheroid; bash -e pipeline_execution.sh

EXPERIMENTS=`cat experiments.csv | tail -n +2`
    
for EXP in $EXPERIMENTS
do
    EXP_NAME=`echo $EXP | cut -d',' -f 1`
    EXP_DIR=`echo $EXP | cut -d',' -f 2`
    EXP_COND=`echo $EXP | cut -d',' -f 3`
    DATA_DIR=$CYTOKIT_DATA_DIR/cytokit/mc38-spheroid/$EXP_NAME/$EXP_DIR
    BASE_CONF=$CYTOKIT_REPO_DIR/pub/config/mc38-spheroid
        
    cytokit config editor --base-config-path=$BASE_CONF --output-dir=$DATA_DIR/output \
    set name "$EXP_NAME.$EXP_DIR.$EXP_COND" \
    set environment.path_formats "get_default_path_formats('1_${EXP_DIR}_{tile:05d}_Z{z:03d}_CH{channel:d}.tif')" \
    save_variant v00/config \
    exit

    # Add this dir to python path to pick up custom cytometer implementation
    export PYTHONPATH=$CYTOKIT_REPO_DIR/pub/analysis/mc38-spheroid
    
    for VARIANT in v00
    do
        OUTPUT_DIR=$DATA_DIR/output/$VARIANT
        CONFIG_DIR=$OUTPUT_DIR/config
        
        echo "Processing experiment $EXP_NAME (config = $CONFIG_DIR, condition = $EXP_COND, dir = $EXP_DIR)"
        
        cytokit processor run_all --config-path=$CONFIG_DIR --data-dir=$DATA_DIR/raw --output-dir=$OUTPUT_DIR --py-log-level=DEBUG
        cytokit operator run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR --raw-dir=$DATA_DIR/raw
        cytokit analysis run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
    done
done
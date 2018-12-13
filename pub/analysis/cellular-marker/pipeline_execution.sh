#!/usr/bin/env bash
# bash -e /lab/repos/cell-image-analysis/analysis/experiments/cellular-marker/pipeline_execution.sh

EXPERIMENTS="
20180614_D22_RepA_Tcell_CD4-CD8-DAPI_5by5;dapi;35
20180614_D22_RepB_Tcell_CD4-CD8-DAPI_5by5;dapi;35
20180614_D23_RepA_Tcell_CD4-CD8-DAPI_5by5;dapi;35
20180614_D23_RepB_Tcell_CD4-CD8-DAPI_5by5;dapi;33
20181116-d40-r1-20x-5by5;pha;25
20181116-d40-r2-20x-5by5;pha;25
20181116-d41-r1-20x-5by5;pha;25
20181116-d41-r2-20x-5by5;pha;25
"

for EXP in $EXPERIMENTS
do
    EXP_NAME=`echo $EXP | cut -d';' -f 1`
    EXP_TYPE=`echo $EXP | cut -d';' -f 2`
    EXP_NUMZ=`echo $EXP | cut -d';' -f 3`
    DATA_DIR=$CYTOKIT_DATA_DIR/cytokit/cellular-marker/$EXP_NAME
    BASE_CONF=$CYTOKIT_REPO_DIR/pub/config/cellular-marker/experiment_${EXP_TYPE}.yaml
    
    # Generate configurations for experiment variants
    cytokit config editor --base-config-path=$BASE_CONF --output-dir=$DATA_DIR/output \
    set name $EXP_NAME \
    set acquisition.num_z_planes $EXP_NUMZ \
    save_variant v00/config \
    set processor.args.run_deconvolution True \
    save_variant v01/config \
    exit
    
    # Run processing for each variant of this experiment
    for VARIANT in v00 v01
    do
        OUTPUT_DIR=$DATA_DIR/output/$VARIANT
        CONFIG_DIR=$OUTPUT_DIR/config
        echo "Processing experiment $EXPERIMENT (variant = $VARIANT, config = $CONFIG_DIR)"
        
        cytokit processor run_all --config-path=$CONFIG_DIR --data-dir=$DATA_DIR/raw --output-dir=$OUTPUT_DIR
        cytokit operator run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
        cytokit analysis run_all  --config-path=$CONFIG_DIR --data-dir=$OUTPUT_DIR 
    done
done

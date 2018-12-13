# source $CYTOKIT_ANALYSIS_REPO_DIR/analysis/experiments/cellular-marker/explorer_config.sh; cytokit application run_explorer

export APP_EXP_NAME=20181116-d40-r1-20x-5by5
export APP_EXP_DATA_DIR=$CYTOKIT_DATA_DIR/cytokit/cellular-marker/$APP_EXP_NAME/output/v00
export APP_EXP_CONFIG_PATH=$APP_EXP_DATA_DIR/config/experiment.yaml
export APP_EXTRACT_NAME=best_z_segm
export APP_MONTAGE_NAME=best_z_segm
export APP_MONTAGE_CHANNEL_NAMES="proc_CD4,proc_CD8"
export APP_MONTAGE_CHANNEL_COLORS="red,green"
export APP_MONTAGE_CHANNEL_RANGES="0-200,0-200"
export APP_PORT=8050
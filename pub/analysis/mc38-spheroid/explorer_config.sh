# source $CYTOKIT_ANALYSIS_REPO_DIR/analysis/experiments/spheroid/20190215-mc38-dmso-control/explorer_config.sh; cytokit application run_explorer
export APP_EXP_NAME="20190215-mc38-dmso-control"
export APP_EXP_COND_DIR="XY02"
export APP_EXP_DATA_DIR=$CYTOKIT_DATA_DIR/spheroid/$APP_EXP_NAME/gdoc_samp/$APP_EXP_COND_DIR/output/v00
export APP_EXP_CONFIG_PATH=$APP_EXP_DATA_DIR/config
export APP_EXTRACT_NAME=segm
export APP_MONTAGE_NAME=segm
export APP_PORT=8050

export APP_MONTAGE_CHANNEL_NAMES="proc_BF,proc_LIVE,proc_DEAD,cyto_cell_boundary"
export APP_MONTAGE_CHANNEL_COLORS="gray,green,red,cyan"
export APP_MONTAGE_CHANNEL_RANGES="0-65535,0-65535,0-65535,0-1"
export APP_MONTAGE_POINT_COLOR="blue"
export APP_GRAPH_POINT_SIZE=8
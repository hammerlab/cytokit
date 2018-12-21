# source $CYTOKIT_ANALYSIS_REPO_DIR/analysis/experiments/20180101_codex_mouse_spleen/explorer_config.sh; python $CYTOKIT_REPO_DIR/python/applications/cytokit_app/explorer/app.py
export APP_EXP_NAME="20180101_codex_mouse_spleen_balbc_slide1"
export APP_EXP_CONFIG_PATH=$CYTOKIT_ANALYSIS_REPO_DIR/config/experiment/20180101_codex_spleen/$APP_EXP_NAME/experiment.yaml
export APP_EXP_DATA_DIR=$CYTOKIT_DATA_DIR/20180101_codex_spleen/$APP_EXP_NAME/output/v00

export APP_MONTAGE_GRID_ENABLED="true"
export APP_MONTAGE_GRID_COLOR="rgba(0, 256, 0, .3)"
# export APP_EXTRACT_NAME=figure2
# export APP_MONTAGE_NAME=figure2
# export APP_MONTAGE_CHANNEL_NAMES="proc_CD11c,proc_CD4"
# export APP_MONTAGE_CHANNEL_COLORS="green,red"

export APP_EXTRACT_NAME=figureS4
export APP_MONTAGE_NAME=figureS4
export APP_MONTAGE_CHANNEL_NAMES="proc_IgD,proc_CD90"
export APP_MONTAGE_CHANNEL_COLORS="green,red"
export APP_MONTAGE_CHANNEL_RANGES="0-22000,0-32000"
export APP_MONTAGE_POINT_COLOR="white"

export APP_CELL_MARKER_MODE="mask"
export APP_CELL_MARKER_MASK_COLOR="rgba(31, 119, 180, 1.0)" # blue
export APP_CELL_MARKER_MASK_FILL="rgba(31, 119, 180, .5)"
export APP_CELL_MARKER_MASK_OBJECT="cell"
export APP_PORT=8050
from codex.ops import op as codex_op
from codex import io as codex_io


class IlluminationCorrection(codex_op.CodexOp):

    def __init__(self, config):
        super().__init__(config)

    def initialize(self):
        pass

    def _run(self, tile, cyto_data=None):
        if cyto_data is None:
            raise ValueError('Cytometry data must be provided for illumination correction')

        # Get configured fields to do range filtering on
        # Compute percentile low and high for each field
        # Save thresholds to monitor
        # Use thresholds to restrict cyto df
        # Retrieve configured channel name to use as target for model
        # Regress channel value / mean against x and y positions
        # Predict illumination image
        # Apply image to tile
        pass
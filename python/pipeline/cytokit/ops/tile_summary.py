from cytokit.ops.op import CytokitOp


class CytokitTileSummary(CytokitOp):

    def __init__(self, config):
        """Operation to record summary statistics for each tile"""
        super().__init__(config)

    def _run(self, tile, **kwargs):
        ncyc, nch, chnames = self.config.n_cycles, self.config.n_channels_per_cycle, self.config.channel_names
        if len(chnames) != ncyc * nch:
            raise ValueError(
                'Number of channel names ({}) does not match num cycles * num channels ({} x {} = {})'
                .format(len(chnames), ncyc, nch, ncyc * nch)
            )
        chnames = iter(chnames)
        for icyc in range(ncyc):
            for ich in range(nch):
                # Tile should have shape (cycles, z, channel, height, width)
                t = tile[icyc, :, ich, :, :]
                self.record({
                    'channel': next(chnames),
                    'mean': t.mean(),
                    'std': t.std(),
                    'min': t.min(),
                    'max': t.max(),
                })
        return tile
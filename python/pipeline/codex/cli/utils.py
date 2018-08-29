#!/usr/bin/python
"""Utility CLI Application

This CLI is helpful for managing CODEX metadata
"""
import fire


class CodexUtils(object):

    def gen_tile_map(self, config_dir):
        """Generate tileMap.txt file for use by CODEXSegm application
        
        Args:
            config_dir: Directory containing experiment configuration files
        Returns:
            Content of tile map as a DataFrame
        """
        import pandas as pd
        from codex import config as codex_config
        config = codex_config.load(config_dir)

        regions = config.region_indexes
        nrow, ncol = config.region_height, config.region_width
        tw, th = config.tile_width, config.tile_height

        df = []
        for reg_idx in regions:
            for tile_x in range(ncol):
                for tile_y in range(nrow):
                    df.append((reg_idx + 1, tile_x + 1, tile_y + 1, tile_x * tw, tile_y * th))
        return pd.DataFrame(df, columns=['RegionNumber', 'TileX', 'TileY', 'Xposition', 'Yposition'])


if __name__ == '__main__':
    fire.Fire(CodexUtils)

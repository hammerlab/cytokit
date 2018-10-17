#!/usr/bin/python
"""Processing pipeline CLI application"""
import fire
import logging
from cytokit.exec import pipeline
from cytokit import cli


class Processor(cli.DataCLI):

    def _get_function_configs(self):
        return [{self.run.__name__: self.config.processor_args}]

    def run(self,
            output_dir,

            # Data subsets to process
            region_indexes=None,
            tile_indexes=None,

            # Execution parameters
            n_workers=None,
            gpus=None,
            memory_limit=48e9,
            tile_prefetch_capacity=1,

            # Processing flags
            run_tile_generator=True,
            run_crop=True,
            run_deconvolution=False,
            run_best_focus=False,
            run_drift_comp=False,
            run_summary=False,
            run_cytometry=False,
            run_illumination_correction=False,
            run_spectral_unmixing=False,

            # Bookkeeping
            record_execution=True):
        """Run processing and cytometry pipeline

        This application can execute the following operations on either raw or already processed data:
            - Drift compensation
            - Deconvolution
            - Selection of best focal planes within z-stacks
            - Cropping of tile overlap
            - Cell segmentation and quantification
            - Illumination correction
            - Spectral Unmixing

        Nothing beyond an input data directory and an output directory are required (see arguments
        below), but GPU information should be provided via the `gpus` argument to ensure that
        all present devices are utilized.  Otherwise, all arguments have reasonable defaults that
        should only need to be changed in special scenarios.

        Args:
            output_dir: Directory to save results in; will be created if it does not exist
            region_indexes: 1-based sequence of region indexes to process; can be specified as:
                - None: Region indexes will be inferred from experiment configuration
                - str or int: A single value will be interpreted as a single index 
                - tuple: A two-item tuple will be interpreted as a right-open range (e.g. '(1,4)' --> [1, 2, 3]) 
                - list: A list of integers will be used as is
            tile_indexes: 1-based sequence of tile indexes to process; has same semantics as `region_indexes`
            n_workers: Number of tiles to process in parallel; should generally match number of gpus and if
                the `gpus` argument is given, then the length of that list will be used as a default (otherwise
                default is 1)
            gpus: 0-based list of gpu indexes to use for processing; has same semantics as other integer
                list arguments like `region_indexes` and `tile_indexes` (i.e. can be a scalar, list, or 2-tuple)
            memory_limit: Maximum amount of memory to allow per-worker; defaults to 48G
            tile_prefetch_capacity: Number of input tiles to buffer into memory for processing; default is 1
                which is nearly always good as this means one tile will undergo processing while a second
                is buffered into memory asynchronously
            run_tile_generator: Flag indicating whether or not the source data to be processed is from un-assembled
                single images (typically raw microscope images) or from already assembled tiles (which would be the
                case if this pipeline has already been run once on raw source data)
            run_crop: Flag indicating whether or not overlapping pixels in raw images should be cropped off; this
                should generally only apply to raw images but will have no effect if images already appear to be
                cropped (though an annoying warning will be printed in that case so this should be set to False
                if not running on raw images with overlap)
            run_deconvolution: Flag indicating whether or not to run deconvolution
            run_best_focus: Flag indicating that best focal plan selection operations should be executed
            run_drift_comp: Flag indicating that drift compensation should be executed
            run_summary: Flag indicating that tile summary statistics should be computed (eg mean, max, min, etc)
            run_cytometry: Flag indicating whether or not image tiles should be segmented and quantified
            run_illumination_correction: Flag indicating whether or not image tiles and cytometry data should be
                adjusted according to global illumination patterns across entire regions
            run_spectral_unmixing: Flag indicating whether or not cross-talk between fluorescent channels should
                be corrected via blind spectral unmixing
            record_execution: Flag indicating whether or not to store arguments and environment in
                a file within the output directory; defaults to True
            record_data: Flag indicating whether or not summary information from each operation
                performed should be included within a file in the output directory; defaults to True
        """
        # Save a record of execution environment and arguments
        if record_execution:
            path = cli.record_execution(output_dir)
            logging.info('Execution arguments and environment saved to "%s"', path)

        # Resolve arguments with multiple supported forms
        region_indexes = cli.resolve_index_list_arg(region_indexes)
        tile_indexes = cli.resolve_index_list_arg(tile_indexes)
        gpus = cli.resolve_int_list_arg(gpus)

        # Set other dynamic defaults
        if n_workers is None:
            # Default to 1 worker given no knowledge of available gpus 
            n_workers = len(gpus) if gpus is not None else 1

        # Configure and run pipeline
        op_flags = pipeline.OpFlags(
            run_crop=run_crop,
            run_deconvolution=run_deconvolution,
            run_best_focus=run_best_focus,
            run_drift_comp=run_drift_comp,
            run_summary=run_summary,
            run_tile_generator=run_tile_generator,
            run_cytometry=run_cytometry,
            run_illumination_correction=run_illumination_correction,
            run_spectral_unmixing=run_spectral_unmixing
        )
        pl_config = pipeline.PipelineConfig(
            self.config, region_indexes, tile_indexes, self.data_dir, output_dir,
            n_workers, gpus, memory_limit, op_flags,
            tile_prefetch_capacity=tile_prefetch_capacity
        )
        pipeline.run(pl_config, logging_init_fn=self._logging_init_fn)


if __name__ == '__main__':
    fire.Fire(Processor)

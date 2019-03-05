# Wrapper class for Ilastik command line execution
import os
import os.path as osp
import subprocess

ILASTIK_HOME = os.getenv('ILASTIK_HOME', '/lab/apps/ilastik')
ILASTIK_RUN = os.getenv('ILASTIK_RUN_SCRIPT', osp.join(ILASTIK_HOME, 'run_ilastik.sh'))


class IlastikError(Exception):
    pass


def to_args(kwargs):
    return ['--{}{}'.format(k, '=' + str(v) if v is not None else '') for k, v in kwargs.items()]


class Ilastik(object):

    def __init__(self, script=ILASTIK_RUN):
        self.script = script

    def run(self, args):
        pres = subprocess.run([self.script] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pres.returncode != 0:
            raise IlastikError(
                'Command {} failed with exit code {}.\nstdout: {}\nstderr:\n{}'
                .format(' '.join(pres.args), pres.returncode, pres.stdout.decode('utf-8'), pres.stderr.decode('utf-8'))
            )
        return pres

    def classify(self, project_path, input_path, output_path, **kwargs):
        """Classify pixels for a single image"""
        # Delete the result if it already exists
        if osp.exists(output_path):
            os.remove(output_path)

        # Run the CLI command
        pres = self.classify_batch(
            project_path, input_path, output_dir='',
            output_filename_format=output_path, **kwargs
        )

        # Verify that the given output path exists
        if not osp.exists(output_path):
            raise ck_ilastik.IlastikError(
                'Ilastik failed to produce result at "{}".\ncommand: {}\nstdout: {}\nstderr:\n{}',
                output_path,
                ' '.join(pres.args),
                pres.stdout.decode('utf-8'),
                pres.stderr.decode('utf-8')
            )
        return pres

    def classify_batch(
            self, project_path, file_pattern, output_dir,
            prediction_type='Simple Segmentation', **kwargs):
        """Classify pixels for multiple images"""
        args = dict(
            headless=None,
            output_format='tif',
            export_source=prediction_type,
            export_dtype='float32',
            project=project_path,
            output_filename_format=osp.join(output_dir, '{nickname}.tif'),
        )
        args.update(kwargs)
        args = to_args(args) + [file_pattern]
        return self.run(args)


CLI = Ilastik()

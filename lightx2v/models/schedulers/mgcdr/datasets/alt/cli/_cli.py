import subprocess
import click

from . import _pkl
from . import _vis
from . import _file


@click.group()
def cli():
    pass


cli.add_command(_pkl.cli, name='pkl')
cli.add_command(_vis.cli, name="vis")
cli.add_command(_file.cli, name="file")


@cli.command('env')
def show_env():
    """ALT Base Environment information"""
    import alt
    msg = f"""
    package : {alt.__file__}
    version : {alt.version}
    author  : qiaolei@senseauto.com

    If you have any new needs, please contact us!!!
    """
    print(msg)


@cli.command()
def update():
    """update alt to the newest version(default branch: main)"""
    update_cmd = 'pip install --trusted-host pypi.kestrel.sensetime.com --index-url http://pypi.kestrel.sensetime.com/simple/ alt --upgrade'  # noqa
    subprocess.run(update_cmd, shell=True)


def main():
    cli()


if __name__ == '__main__':
    main()

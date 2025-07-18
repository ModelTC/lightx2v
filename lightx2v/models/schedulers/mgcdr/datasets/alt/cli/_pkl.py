import click
from alt.utils import dump

from alt.modules.pkl_module import AltPklModule as APM


@click.group()
@click.pass_context
def cli(ctx):
    """Provide the client about Ceph and local pickle"""
    pass


@cli.command('tj')
@click.argument('input')
@click.argument('output')
def tj(input, output):
    """the client transfer pkl(ceph&local) to json(local)"""
    amp = APM(input)
    click.secho("{} -> {}".format(input, output), fg="green")
    dump(output, amp.json_data)


@cli.command('cat')
@click.argument('input')
def cat(input):
    """the client cat pkl(ceph&local), equal to print"""
    APM(input).cat()


@cli.command("lookup")
@click.argument('input')
@click.option("-k", "--key", default="", help="your lookup key, frames[0]")
@click.option("-t", "--timestamp", default="", help="select timestamp 1710242168166666.2")
def lookup(input, key, timestamp):
    """the client lookup information from alt pkl(ceph&local)"""
    APM(input).lookup(key=key, timestamp=timestamp)


if __name__ == '__main__':
    cli()

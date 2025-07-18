import click
from loguru import logger
from alt.visualize.object import Visualize, BEVVisualizer
from alt.utils.petrel_helper import petreloss_path
from alt.utils.file_helper import dump


@click.group()
@click.pass_context
def cli(ctx):
    """Provide the client about autolabel objects visualize"""
    pass


@cli.command('object')
@click.argument('meta_path', type=str)
@click.argument('camera_name', type=str)
@click.option('--show', type=bool, default=True, help='Displays the visualization in a window.')
@click.option('--vis_box2d', type=bool, default=True, help='Whether to view 2D bounding boxes.')
@click.option('--vis_box3d', type=bool, default=True, help='Whether to view 3D bounding boxes.')
@click.option('--vis_point_cloud', type=bool, default=False, help='Whether to view point cloud.')
@click.option('--vis_point_cloud_3d', type=bool, default=False, help='view 3D boxes in point cloud.')
@click.option('--save_video', type=bool, default=True, help='Whether to save video.')
@click.option('--save_img', type=bool, default=True, help='Whether to save image.')
@click.option('--save_path', type=str, default=None, help='The path to save the video or image.')
@click.option(
    '--wait_time', type=int, default=30, help=('Display a window for given milliseconds or until any key is pressed.')
)
@click.option('--fps', type=int, default=10, help='The frame rate of the video to save.')
@click.option(
    '--lidar_side',
    type=int,
    default=80,
    help=('Lidar side determines the horizontal viewing range of the BEV, in meters.'),
)
@click.option(
    '--lidar_fwd',
    type=int,
    default=120,
    help=('Lidar forward determines the vertical viewing range of the BEV, in meters.'),
)
@click.option(
    '--rotation_bev',
    type=int,
    default=0,
    help=(
        'Rotate the BEV so that the ego vehicle is facing upwards. '
        'The default value of 0 is generally used, but GAC-10 needs to set the value to 90.'
    ),
)
def vis_objects(
    meta_path,
    camera_name,
    show,
    vis_box2d,
    vis_box3d,
    vis_point_cloud,
    vis_point_cloud_3d,
    save_video,
    save_img,
    save_path,
    wait_time,
    fps,
    lidar_side,
    lidar_fwd,
    rotation_bev,
):
    """the client transfer pkl(ceph&local) to json(local)"""

    logger.info('Start')
    visualize = Visualize(meta_path=meta_path, camera_name=camera_name, conf_path=petreloss_path())
    lidar_side = 80
    lidar_fwd = 120
    visualize.vis_camera_and_BEV(
        show=show,
        vis_box2d=vis_box2d,
        vis_box3d=vis_box3d,
        vis_point_cloud=vis_point_cloud,
        vis_point_cloud_3d=vis_point_cloud_3d,
        save_video=save_video,
        save_img=save_img,
        save_path=save_path,
        wait_time=wait_time,
        fps=fps,
        lidar_side_range=(-lidar_side, lidar_side),
        lidar_fwd_range=(-lidar_fwd, lidar_fwd),
        rotation_bev=rotation_bev,
    )
    logger.info('End!')


@cli.command('bev')
@click.argument('meta_path', type=str)
@click.argument("label_path", type=str)
@click.option('--box2d_threshold', type=float, default=0.05, help='box2d_threshold.')
@click.option('--box3d_threshold', type=float, default=0.05, help='box3d_threshold.')
@click.option('--lidar_side_range', type=str, default="-50,50", help='lidar_side_range.')
@click.option('--lidar_fwd_range', type=str, default="-100,200", help='lidar_side_range.')
@click.option('--num_horizontal_bins', type=int, default=20, help='num_horizontal_bins, freeze is better')
@click.option('--save_path', default=None, help='img save_path')
def vis_bev_objects(
    meta_path,
    label_path,
    box2d_threshold,
    box3d_threshold,
    lidar_side_range,
    lidar_fwd_range,
    num_horizontal_bins,
    save_path,
):
    """the client vis multi camera"""

    logger.info('Start')
    viser = BEVVisualizer(meta_json=meta_path,
                          box2d_threshold=box2d_threshold,
                          box3d_threshold=box3d_threshold,
                          lidar_side_range=[float(item) for item in lidar_side_range.split(",")],
                          lidar_fwd_range=[float(item) for item in lidar_fwd_range.split(",")],
                          num_horizontal_bins=num_horizontal_bins,
                          save_path=save_path)
    res = viser.process()

    dump(label_path, res)
    logger.info('End!')


if __name__ == '__main__':
    cli()

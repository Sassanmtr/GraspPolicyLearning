from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})
import omni.client
import omni.kit
import omni.usd
import asyncio
import os
from pathlib import Path

# Code to convert SHapeNet models to USD

def file_exists_on_omni(file_path):
    result, _ = omni.client.stat(file_path)
    if result == omni.client.Result.OK:
        return True

    return False


async def create_folder_on_omni(folder_path):
    if not file_exists_on_omni(folder_path):
        result = await omni.client.create_folder_async(folder_path)
        return result == omni.client.Result.OK


async def convert(in_file, out_file):
    # Folders must be created first through usd_ext of omni won't be able to create the files creted in them in the current session.
    out_folder = out_file[0 : out_file.rfind("/") + 1]

    # only call create_folder_on_omni if it's connected to an omni server
    if out_file.startswith("omniverse://"):
        await create_folder_on_omni(out_folder + "materials")

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.as_shapenet = True
    converter_context.single_mesh = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file, out_file, progress_callback, converter_context
    )

    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def ShapePrim(input_path, output_path):
    # omni_path = "test3" + ".usd"
    # output_path = output_dir + omni_path
    print("---Converting...")
    status = asyncio.get_event_loop().run_until_complete(
        convert(input_path, output_path)
    )
    if not status:
        return f"ERROR OmniConverterStatus is {status}"

directory = Path.cwd() / "usd_meshes" 

for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".obj"):
                obj_path = os.path.join(root, file)
                h5_path = os.path.join(root, file[:-4] + ".h5")
                output_path = os.path.join(directory, os.path.relpath(obj_path, directory))
                output_path = os.path.splitext(output_path)[0] + ".usd"
                ShapePrim(obj_path, output_path)
                print("{} converted to {}".format(obj_path, output_path))

print("done!")
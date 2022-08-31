from utils import save_images_and_traj_for_ngp
from optitrack_tools.endoscope_definitions import ENDOSCOPES

directory = './test_data'

save_images_and_traj_for_ngp(directory,
                             './test_output',
                             endoscope=ENDOSCOPES.ITO,
                             frame_samples=150,
                             runtime_stops=(10, 20),
                             )

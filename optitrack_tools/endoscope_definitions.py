import numpy as np
from dataclasses import dataclass


@dataclass
class Endoscope:
    """
    A dataclass holding the data needed for reconstructing the pose of an endoscope using optitrack markers.
    """
    cad_balls: np.ndarray = None
    """The location of the markers from the markers' CG to each marker. In CAD coordinates. """
    light_2_balls: np.ndarray = None
    """ The vector from the intersection of the endoscope shaft and the light post to the marker's CG. In CAD coord."""
    geometric_cg: np.ndarray = None
    """ The location of the markers' CG in CAD coordinates. """
    shaft: np.ndarray = None
    """ The length of the endoscope shaft **from** tip to light post. In CAD coord. """
    angle: float = None
    """ The tip angle of the endoscope in degrees. """
    stl_name: str = None
    """ The name of the STL file for loading the CAD Geometry """
    rigid_body_name: str = None
    """ The name given to the rigid body in optitrack. """


class ENDOSCOPES:
    """ A namespace for the different endoscopes we can use """
    ITO = Endoscope(cad_balls=np.array([[29.66957, 47.6897, 83.68694],
                                        [29.66957, -64.42155, 83.68694],
                                        [-70.6104, -48.7631, -69.62806],
                                        [11.27126, 65.49496, -97.74583]]),
                    light_2_balls=np.array([-24.9378, 8.3659, 13.40425]),
                    geometric_cg=np.array([-9.43783, 8.36593, 13.40425]),
                    shaft=np.array([321.5, 0, 0]),
                    angle=30,
                    stl_name='endoscope.stl',
                    rigid_body_name='endo-front')

    TUEBINGEN = Endoscope(cad_balls=np.array([[-15.67487, 50.68646, 57.54749],
                                              [-38.36799, -59.62912, 19.26329],
                                              [46.15406, -46.74515, -1.61367],
                                              [-7.8888, 99.25303, -35.45885]]),
                          light_2_balls=np.array([-16.10914, 5.29419, 1.30581]),
                          geometric_cg=np.array([-16.10914, 5.29419, 5.59711]),
                          shaft=np.array([0, 0, 315]),
                          angle=70,
                          stl_name='endoskop_2_assy',
                          rigid_body_name='endo-front-2')

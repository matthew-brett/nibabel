# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Read / write access to NIfTI2 image format

Format described here:

    http://www.nitrc.org/forum/message.php?msg_id=3738

Stuff about the CIFTI2 file format here:

    http://www.nitrc.org/plugins/mwiki/index.php/cifti2:ConnectivityMatrixFileFormats

'''
from __future__ import division, print_function, absolute_import

import numpy as np

from .. import xmlutils as xml
from ..externals import inflection
from ..externals.six import string_types
from ..filebasedimages import FileBasedHeader, FileBasedImage
from ..nifti2 import Nifti2Image


CIFTI_MAP_TYPES = ('CIFTI_INDEX_TYPE_BRAIN_MODELS',
                   'CIFTI_INDEX_TYPE_PARCELS',
                   'CIFTI_INDEX_TYPE_SERIES',
                   'CIFTI_INDEX_TYPE_SCALARS',
                   'CIFTI_INDEX_TYPE_LABELS')

CIFTI_MODEL_TYPES = ('CIFTI_MODEL_TYPE_SURFACE',
                     'CIFTI_MODEL_TYPE_VOXELS')

CIFTI_SERIESUNIT_TYPES = ('SECOND',
                          'HERTZ',
                          'METER',
                          'RADIAN')

CIFTI_BrainStructures = ('CIFTI_STRUCTURE_ACCUMBENS_LEFT',
                         'CIFTI_STRUCTURE_ACCUMBENS_RIGHT',
                         'CIFTI_STRUCTURE_ALL_WHITE_MATTER',
                         'CIFTI_STRUCTURE_ALL_GREY_MATTER',
                         'CIFTI_STRUCTURE_AMYGDALA_LEFT',
                         'CIFTI_STRUCTURE_AMYGDALA_RIGHT',
                         'CIFTI_STRUCTURE_BRAIN_STEM',
                         'CIFTI_STRUCTURE_CAUDATE_LEFT',
                         'CIFTI_STRUCTURE_CAUDATE_RIGHT',
                         'CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_LEFT',
                         'CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_RIGHT',
                         'CIFTI_STRUCTURE_CEREBELLUM',
                         'CIFTI_STRUCTURE_CEREBELLUM_LEFT',
                         'CIFTI_STRUCTURE_CEREBELLUM_RIGHT',
                         'CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_LEFT',
                         'CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_RIGHT',
                         'CIFTI_STRUCTURE_CORTEX',
                         'CIFTI_STRUCTURE_CORTEX_LEFT',
                         'CIFTI_STRUCTURE_CORTEX_RIGHT',
                         'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT',
                         'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT',
                         'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT',
                         'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT',
                         'CIFTI_STRUCTURE_OTHER',
                         'CIFTI_STRUCTURE_OTHER_GREY_MATTER',
                         'CIFTI_STRUCTURE_OTHER_WHITE_MATTER',
                         'CIFTI_STRUCTURE_PALLIDUM_LEFT',
                         'CIFTI_STRUCTURE_PALLIDUM_RIGHT',
                         'CIFTI_STRUCTURE_PUTAMEN_LEFT',
                         'CIFTI_STRUCTURE_PUTAMEN_RIGHT',
                         'CIFTI_STRUCTURE_THALAMUS_LEFT',
                         'CIFTI_STRUCTURE_THALAMUS_RIGHT')


def _value_if_klass(val, klass, none_ok=True):
    if none_ok and val is None:
        return val
    elif isinstance(val, klass):
        return val
    else:
        raise ValueError('Not a valid %s instance.' % klass.__name__)


def _value_or_make_klass(val, klass):
    if val is None:
        return klass()
    else:
        return _value_if_klass(val, klass)


class Cifti2MetaData(xml.XmlSerializable):
    """ A list of key-value pairs stored in the list self.data """

    def __init__(self, nvpair=None):
        self.data = []
        self.add_metadata(nvpair)

    def _add_remove_metadata(self, metadata, func):
        pairs = []
        if isinstance(metadata, dict):
            pairs = metadata.items()
        elif isinstance(metadata, (list, tuple)):
            if not isinstance(metadata[0], string_types):
                for item in metadata:
                    self._add_remove_metadata(item, func)
                return
            elif len(metadata) == 2:
                pairs = [tuple((metadata[0], metadata[1]))]
            else:
                raise ValueError('nvpair must be a 2-list or 2-tuple')
        else:
            raise ValueError('nvpair input must be a list, tuple or dict')

        for pair in pairs:
            if func == 'add':
                if pair not in self.data:
                    self.data.append(pair)
            elif func == 'remove':
                self.data.remove(pair)
            else:
                raise ValueError('Unknown func %s' % func)

    def add_metadata(self, metadata):
        """Add metadata key-value pairs

        This allows storing multiple keys with the same name but different
        values.


        Parameters
        ----------
        metadata : 2-List, 2-Tuple, Dictionary, List[2-List or 2-Tuple]
                     Tuple[2-List or 2-Tuple]

        Returns
        -------
        None

        """
        if metadata is None:
            return
        self._add_remove_metadata(metadata, 'add')

    def remove_metadata(self, metadata):
        if metadata is None:
            return
        self._add_remove_metadata(metadata, 'remove')

    def _to_xml_element(self):
        metadata = xml.Element('MetaData')

        for name_text, value_text in self.data:
            md = xml.SubElement(metadata, 'MD')
            name = xml.SubElement(md, 'Name')
            name.text = str(name_text)
            value = xml.SubElement(md, 'Value')
            assert(value_text is not None)
            value.text = str(value_text)
        return metadata


class Cifti2LabelTable(xml.XmlSerializable):

    def __init__(self):
        self.labels = []

    @property
    def num_labels(self):
        return len(self.labels)

    def get_labels_as_dict(self):
        self.labels_as_dict = {}
        for ele in self.labels:
            self.labels_as_dict[ele.key] = ele.label
        return self.labels_as_dict

    def _to_xml_element(self):
        assert len(self.labels) > 0
        labeltable = xml.Element('LabelTable')
        for ele in self.labels:
            labeltable.append(ele._to_xml_element())
        return labeltable

    def print_summary(self):
        print(self.get_labels_as_dict())


class Cifti2Label(xml.XmlSerializable):

    def __init__(self, key=0, label='', red=None,
                 green=None, blue=None, alpha=None):
        self.key = key
        self.label = label
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    @property
    def rgba(self):
        """ Returns RGBA as tuple """
        return (self.red, self.green, self.blue, self.alpha)

    def _to_xml_element(self):
        lab = xml.SubElement(labeltable, 'Label')
        lab.attrib['Key'] = str(ele.key)
        lab.text = str(ele.label)
        if ele.red is not None:
            lab.attrib['Red'] = str(ele.red)
        if ele.green is not None:
            lab.attrib['Green'] = str(ele.green)
        if ele.blue is not None:
            lab.attrib['Blue'] = str(ele.blue)
        if ele.alpha is not None:
            lab.attrib['Alpha'] = str(ele.alpha)
        return lab


class Cifti2NamedMap(xml.XmlSerializable):
    """Class for Named Map"""
    map_name = str

    def __init__(self, map_name=None, metadata=None, label_table=None):
        self.map_name = map_name
        self.metadata = metadata  # _value_or_make_klass(metadata, Cifti2MetaData)
        self.label_table = label_table  # _value_or_make_klass(label_table, Cifti2LabelTable)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """ Set the metadata for this NamedMap

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        self._metadata = _value_if_klass(metadata, Cifti2MetaData)

    @property
    def label_table(self):
        return self._label_table

    @label_table.setter
    def label_table(self, label_table):
        """ Set the label_table for this NamedMap

        Parameters
        ----------
        label_table : Cifti2LabelTable

        Returns
        -------
        None
        """
        self._label_table = _value_if_klass(label_table, Cifti2LabelTable)

    def _to_xml_element(self):
        named_map = xml.Element('NamedMap')
        if self.metadata:
            named_map.append(self.metadata._to_xml_element())
        if self.label_table:
            named_map.append(self.label_table._to_xml_element())
        map_name = xml.SubElement(named_map, 'MapName')
        map_name.text = self.map_name
        return named_map


class Cifti2Surface(xml.XmlSerializable):
    """Class for Surface """
    # brainStructure = str
    # surfaceNumberOfVertices = int

    def __init__(self, brain_structure=None, surface_number_of_vertices=None):
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = surface_number_of_vertices

    def _to_xml_element(self):
        assert self.brain_structure is not None
        surf = xml.Element('Surface')
        surf.attrib['BrainStructure'] = str(self.brain_structure)
        surf.attrib['SurfaceNumberOfVertices'] = str(self.surface_number_of_vertices)
        return surf


class Cifti2VoxelIndicesIJK(xml.XmlSerializable):
    # indices = np.array

    def __init__(self, indices=None):
        self.indices = indices

    def _to_xml_element(self):
        assert self.indices is not None
        vox_ind = xml.Element('VoxelIndicesIJK')
        if self.indices is not None:
            vox_ind.text = ''
            for row in self.indices:
                vox_ind.text += ' '.join(row.astype(str).tolist()) + '\n'
        return vox_ind


class Cifti2Vertices(xml.XmlSerializable):

    # brain_structure = str
    # vertices = np.array

    def __init__(self, brain_structure=None, vertices=None):
        self.vertices = vertices
        self.brain_structure = brain_structure

    def _to_xml_element(self):
        assert self.vertices is not None
        vertices = xml.Element('Vertices')
        vertices.attrib['BrainStructure'] = str(self.brain_structure)

        if self.vertices is not None:
            vertices.text = ' '.join(self.vertices.astype(str).tolist())
        return vertices


class Cifti2Parcel(object):
    """Class for Parcel"""
    # name = str

    def __init__(self, name=None, voxel_indices_ijk=None, vertices=None):
        self.name = name
        self.voxel_indices_ijk = voxel_indices_ijk
        self.vertices = vertices if vertices is not None else []

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    def add_cifti_vertices(self, vertices):
        """ Adds a vertices to the Cifti2Parcel

        Parameters
        ----------
        vertices : Cifti2Vertices
        """
        if not isinstance(vertices, Cifti2Vertices):
            raise TypeError("Not a valid Cifti2Vertices instance")
        self.vertices.append(vertices)

    def remove_cifti2_vertices(self, ith):
        """ Removes the ith vertices element from the Cifti2Parcel """
        self.vertices.pop(ith)

    def _to_xml_element(self):
        assert self.name is not None
        parcel = xml.Element('Parcel')
        parcel.attrib['Name'] = str(self.name)
        if self.voxel_indices_ijk:
            parcel.append(self.voxel_indices_ijk._to_xml_element())
        for vertex in self.vertices:
            parcel.append(vertex._to_xml_element())
        return parcel


class Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(object):

    # meterExponent = int
    # matrix = np.array

    def __init__(self, meter_exponent=None, matrix=None):
        self.meter_exponent = meter_exponent
        self.matrix = matrix

    def _to_xml_element(self):
        assert self.matrix is not None
        trans = xml.Element('TransformationMatrixVoxelIndicesIJKtoXYZ')
        trans.attrib['MeterExponent'] = str(self.meter_exponent)
        if self.matrix is not None:
            trans.text = ''
            for row in self.matrix:
                trans.text += ' '.join(['%.10f' % val for val in row]) + "\n"
        return trans


class Cifti2Volume(object):

    # volumeDimensions = np.array
    # transformationMatrixVoxelIndicesIJKtoXYZ = np.array

    def __init__(self, volume_dimensions=None, transform_matrix=None):
        self.volume_dimensions = volume_dimensions
        self.transformation_matrix_voxel_indices_ijk_to_xyz = transform_matrix

    def _to_xml_element(self):
        assert self.volume_dimensions is not None
        volume = xml.Element('Volume')
        volume.attrib['VolumeDimensions'] = ','.join(
            [str(val) for val in self.volume_dimensions])
        volume.append(self.transformation_matrix_voxel_indices_ijk_to_xyz._to_xml_element())
        return volume


class Cifti2VertexIndices(object):
    # indices = np.array

    def __init__(self, indices=None):
        self.indices = indices

    def _to_xml_element(self):
        assert self.indices is not None
        vert_indices = xml.Element('VertexIndices')
        vert_indices.text = ' '.join(self.indices.astype(str).tolist())
        return vert_indices


class Cifti2BrainModel(object):

    # index_offset = int
    # index_count = int
    # model_type = str
    # brain_structure = str
    # surface_number_of_vertices = int
    # voxel_indices_ijk = np.array
    # vertex_indices = np.array

    def __init__(self, index_offset=None, index_count=None, model_type=None,
                 brain_structure=None, n_surface_vertices=None,
                 voxel_indices_ijk=None, vertex_indices=None):
        self.index_offset = index_offset
        self.index_count = index_count
        self.model_type = model_type
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = n_surface_vertices

        self.voxel_indices_ijk = voxel_indices_ijk
        self.vertex_indices = vertex_indices

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    @property
    def vertex_indices(self):
        return self._vertex_indices

    @vertex_indices.setter
    def vertex_indices(self, value):
        self._vertex_indices = _value_if_klass(value, Cifti2VertexIndices)

    def _to_xml_element(self):
        brain_model = xml.Element('BrainModel')

        for key in ['IndexOffset', 'IndexCount', 'ModelType', 'BrainStructure',
                    'SurfaceNumberOfVertices']:
            attr = inflection.underscore(key)
            value = getattr(self, attr)
            if value is not None:
                brain_model.attrib[key] = str(value)
        if self.voxel_indices_ijk:
            brain_model.append(self.voxel_indices_ijk._to_xml_element())
        if self.vertex_indices:
            brain_model.append(self.vertex_indices._to_xml_element())
        return brain_model


class Cifti2MatrixIndicesMap(object):
    """Class for Matrix Indices Map

    Provides a mapping between matrix indices and their interpretation.
    """
    # applies_to_matrix_dimension = int
    # indices_map_to_data_type = str
    # number_of_series_points = int
    # series_exponent = int
    # series_start = float
    # series_step = float
    # series_unit = str

    def __init__(self, applies_to_matrix_dimension,
                 indices_map_to_data_type,
                 number_of_series_points=None,
                 series_exponent=None,
                 series_start=None,
                 series_step=None,
                 series_unit=None,
                 brain_models=None,
                 named_maps=None,
                 parcels=None,
                 surfaces=None,
                 volume=None):
        self.applies_to_matrix_dimension = applies_to_matrix_dimension
        self.indices_map_to_data_type = indices_map_to_data_type
        self.number_of_series_points = number_of_series_points
        self.series_exponent = series_exponent
        self.series_start = series_start
        self.series_step = series_step
        self.series_unit = series_unit
        self.brain_models = brain_models if brain_models is not None else []
        self.named_maps = named_maps if named_maps is not None else []
        self.parcels = parcels if parcels is not None else []
        self.surfaces = surfaces if surfaces is not None else []
        self.volume = volume  # _value_or_make_klass(volume, Cifti2Volume)

    def add_cifti_brain_model(self, brain_model):
        """ Adds a brain model to the Cifti2MatrixIndicesMap

        Parameters
        ----------
        brain_model : Cifti2BrainModel
        """
        if not isinstance(brain_model, Cifti2BrainModel):
            raise TypeError("Not a valid Cifti2BrainModel instance")
        self.brain_models.append(brain_model)

    def remove_cifti_brain_model(self, ith):
        """ Removes the ith brain model element from the Cifti2MatrixIndicesMap """
        self.brain_models.pop(ith)

    def add_cifti_named_map(self, named_map):
        """ Adds a named map to the Cifti2MatrixIndicesMap

        Parameters
        ----------
        named_map : Cifti2MatrixIndicesMap
        """
        if isinstance(named_map, Cifti2MatrixIndicesMap):
            raise TypeError("Not a valid Cifti2MatrixIndicesMap instance")
        self.named_maps.append(named_map)

    def remove_cifti_named_map(self, ith):
        """ Removes the ith named_map element from the Cifti2MatrixIndicesMap """
        self.named_maps.pop(ith)

    def add_cifti_parcel(self, parcel):
        """ Adds a parcel to the Cifti2MatrixIndicesMap

        Parameters
        ----------
        parcel : Cifti2Parcel
        """
        if not isinstance(parcel, Cifti2Parcel):
            raise TypeError("Not a valid Cifti2Parcel instance")
        self.parcels.append(parcel)

    def remove_cifti2_parcel(self, ith):
        """ Removes the ith parcel element from the Cifti2MatrixIndicesMap """
        self.parcels.pop(ith)

    def add_cifti_surface(self, surface):
        """ Adds a surface to the Cifti2MatrixIndicesMap

        Parameters
        ----------
        surface : Cifti2Surface
        """
        if not isinstance(surface, Cifti2Surface):
            raise TypeError("Not a valid Cifti2Surface instance")
        self.surfaces.append(surface)

    def remove_cifti2_surface(self, ith):
        """ Removes the ith surface element from the Cifti2MatrixIndicesMap """
        self.surfaces.pop(ith)

    def set_cifti2_volume(self, volume):
        """ Adds a volume to the Cifti2MatrixIndicesMap

        Parameters
        ----------
        volume : Cifti2Volume
        """
        if not isinstance(volume, Cifti2Volume):
            raise TypeError("Not a valid Cifti2Volume instance")
        self.volume = volume

    def remove_cifti2_volume(self):
        """ Removes the volume element from the Cifti2MatrixIndicesMap """
        self.volume = None

    def _to_xml_element(self):
        assert self.applies_to_matrix_dimension is not None

        mat_ind_map = xml.Element('MatrixIndicesMap')
        for key in ['AppliesToMatrixDimension', 'IndicesMapToDataType',
                    'NumberOfSeriesPoints', 'SeriesExponent', 'SeriesStart',
                    'SeriesStep', 'SeriesUnit']:
            attr = inflection.underscore(key)
            value = getattr(self, attr)
            if value is not None:
                mat_ind_map.attrib[key] = str(value)
        for named_map in self.named_maps:
            assert named_map._to_xml_element() is not None
            mat_ind_map.append(named_map._to_xml_element())
        for surface in self.surfaces:
            assert surface._to_xml_element() is not None
            mat_ind_map.append(surface._to_xml_element())
        for parcel in self.parcels:
            assert parcel._to_xml_element() is not None
            mat_ind_map.append(parcel._to_xml_element())
        if self.volume:
            assert self.volume._to_xml_element() is not None
            mat_ind_map.append(self.volume._to_xml_element())
        for model in self.brain_models:
            assert model._to_xml_element() is not None
            mat_ind_map.append(model._to_xml_element())
        return mat_ind_map


class Cifti2Matrix(xml.XmlSerializable):
    def __init__(self, meta=None, mims=None):
        self.mims = mims if mims is not None else []
        self.metadata = meta

    @property
    def metadata(self):
        return self._meta

    @metadata.setter
    def metadata(self, meta):
        """ Set the metadata for this Cifti2Header

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        if meta is not None and not isinstance(meta, Cifti2MetaData):
            raise TypeError("Not a valid Cifti2MetaData instance")
        self._meta = meta

    def add_cifti_matrix_indices_map(self, mim):
        """ Adds a matrix indices map to the Cifti2Matrix

        Parameters
        ----------
        mim : Cifti2MatrixIndicesMap
        """
        if isinstance(mim, Cifti2MatrixIndicesMap):
            self.mims.append(mim)
        else:
            raise TypeError("Not a valid Cifti2MatrixIndicesMap instance")

    def remove_cifti2_matrix_indices_map(self, ith):
        """ Removes the ith matrix indices map element from the Cifti2Matrix """
        self.mims.pop(ith)

    def _to_xml_element(self):
        assert len(self.mims) != 0 or self.metadata is not None
        mat = xml.Element('Matrix')
        if self.metadata:
            mat.append(self.metadata._to_xml_element())
        for mim in self.mims:
            mat.append(mim._to_xml_element())
        return mat


class Cifti2Header(FileBasedHeader, xml.XmlSerializable):
    ''' Class for Cifti2 header extension '''

    # version = str

    def __init__(self, matrix=None, version="2.0"):
        FileBasedHeader.__init__(self)
        xml.XmlSerializable.__init__(self)
        if matrix is None:
            self.matrix = Cifti2Matrix()
        else:
            self.matrix = matrix
        self.version = version

    def _to_xml_element(self):
        cifti = xml.Element('CIFTI')
        cifti.attrib['Version'] = str(self.version)
        mat_xml = self.matrix._to_xml_element()
        if mat_xml is not None:
            cifti.append(mat_xml)
        return cifti

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            return klass()
        if type(header) == klass:
            return header.copy()
        raise ValueError('header is not a Cifti2Header')

    @classmethod
    def may_contain_header(klass, binaryblock):
        from .parse_cifti2_fast import _Cifti2AsNiftiHeader
        return _Cifti2AsNiftiHeader.may_contain_header(binaryblock)


class Cifti2Image(FileBasedImage):
    # It is a Nifti2Image, but because Nifti2Image object
    # contains both the *format* and the assumption that it's
    # a spatial image, we can't inherit directly.
    header_class = Cifti2Header
    valid_exts = Nifti2Image.valid_exts
    files_types = Nifti2Image.files_types
    makeable = False
    rw = True

    def __init__(self, data=None, header=None, nifti_header=None):
        self._header = header or Cifti2Header()
        self.data = data
        self.extra = nifti_header

    def get_data(self):
        return self.data

    @classmethod
    def from_file_map(klass, file_map):
        """ Load a Cifti2 image from a file_map

        Parameters
        file_map : string

        Returns
        -------
        img : Cifti2Image
            Returns a Cifti2Image
         """
        from .parse_cifti2_fast import _Cifti2AsNiftiImage, Cifti2Extension
        nifti_img = _Cifti2AsNiftiImage.from_file_map(file_map)

        # Get cifti2 header
        cifti_header = reduce(lambda accum, item:
                              item.get_content()
                              if isinstance(item, Cifti2Extension)
                              else accum,
                              nifti_img.get_header().extensions or [],
                              None)
        if cifti_header is None:
            raise ValueError('Nifti2 header does not contain a CIFTI2 '
                             'extension')

        # Construct cifti image
        cifti_img = Cifti2Image(data=np.squeeze(nifti_img.get_data()),
                                header=cifti_header,
                                nifti_header=nifti_img.get_header())
        cifti_img.file_map = nifti_img.file_map
        return cifti_img

    def to_file_map(self, file_map=None):
        """ Save the current image to the specified file_map

        Parameters
        ----------
        file_map : string

        Returns
        -------
        None
        """
        from .parse_cifti2_fast import Cifti2Extension
        header = self.extra
        extension = Cifti2Extension(content=self.header.to_xml().encode())
        header.extensions.append(extension)
        data = np.reshape(self.data, [1, 1, 1, 1] + list(self.data.shape))
        img = Nifti2Image(data, None, header)
        img.to_file_map(file_map or self.file_map)


class Cifti2DenseDataSeriesHeader(Cifti2Header):

    @classmethod
    def may_contain_header(klass, binaryblock):
        from .parse_cifti2_fast import _Cifti2DenseDataSeriesNiftiHeader
        return _Cifti2DenseDataSeriesNiftiHeader.may_contain_header(binaryblock)


class Cifti2DenseDataSeries(Cifti2Image):
    """Class to handle Dense Data Series
    Dense Data Series
    -----------------

    Intent_code: 3002, NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES
    Intent_name: ConnDenseSeries
    File extension: .dtseries.nii
    AppliesToMatrixDimension 0: series
    AppliesToMatrixDimension 1: brain models

    This file type represents data points in a series for every vertex and voxel
    in the mapping.  A row is a complete data series,for a single vertex or
    voxel in the mapping that applies along the second dimension. A data series
    is often a timeseries, but it can also represent other data types such as a
    series of sampling depths along the surface normal from the white to pial
    surface.  It retains the 't' in dtseries from CIFTI-1 naming conventions.
    """
    header_class = Cifti2DenseDataSeriesHeader
    valid_exts = ('.dtseries.nii',)
    files_types = (('image', '.dtseries.nii'),)


def load(filename):
    """ Load cifti2 from `filename`

    Parameters
    ----------
    filename : str
        filename of image to be loaded

    Returns
    -------
    img : Cifti2Image
        cifti image instance

    Raises
    ------
    ImageFileError: if `filename` doesn't look like cifti
    IOError : if `filename` does not exist
    """
    from .parse_cifti2_fast import _Cifti2AsNiftiImage
    return _Cifti2AsNiftiImage.from_filename(filename)


def save(img, filename):
    """ Save cifti to `filename`

    Parameters
    ----------
    filename : str
        filename to which to save image
    """
    Cifti2Image.instance_to_filename(img, filename)
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Reading / writing functions for Brainvoyager (BV) VTC files

for documentation on the file format see:
http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html

Author: Thomas Emmerling
'''

import numpy as np
from .bv import BvError,BvFileHeader,BvFileImage
from .spatialimages import HeaderDataError, HeaderTypeError
from .batteryrunners import Report

def _make_vtc_header_dtd(fmrlt, prts):
    ''' Helper for creating a VTC header dtype with given parameters

    Header format from
    http://support.brainvoyager.com/installation-introduction/23-file-formats/379-users-guide-23-the-format-of-vtc-files.html

    Parameters
    ----------
    fmrlt : int
        length of FMR file name including terminating zero
    prts : sequence
        sequence of ints, giving lengths of PRT filenames, including
        terminating zeros.

    Returns
    -------
    dtd : list
        data type definition, list of 2-tuples where tuples are (field name,
        field type definition).  This can be made into a numpy dtype with
        ``np.dtype(dtd)``
    '''
    if len(prts) == 0:
        prt_fmts = [('prt0', 'S1')]
    else:
        prt_fmts = [('prt{0}'.format(i), 'S{0}'.format(L+1))
                    for i, L in enumerate(prts)]
    vtc_header_dtd = \
        [
            ('version', 'i2'),
            ('fmr', 'S{0}'.format(fmrlt)),
            ('nPrt', 'i2'),
            ('prts', prt_fmts),
            ('currentPrt', 'i2'),
            ('datatype', 'i2'),
            ('volumes', 'i2'),
            ('Resolution', 'i2'),
            ('XStart', 'i2'),
            ('XEnd', 'i2'),
            ('YStart', 'i2'),
            ('YEnd', 'i2'),
            ('ZStart', 'i2'),
            ('ZEnd', 'i2'),
            ('LRConvention', 'i1'),
            ('RefSpace', 'i1'),
            ('TR', 'f4'),
        ]
    return vtc_header_dtd

class VtcHeader(BvFileHeader):

    # format defaults
    allowed_dtypes = [2]
    default_dtype = 2

    def get_data_shape(self):
        ''' Get shape of data
        '''
        hdr = self._structarr
        # calculate dimensions
        z = (hdr['ZEnd'] - hdr['ZStart']) / hdr['Resolution']
        y = (hdr['YEnd'] - hdr['YStart']) / hdr['Resolution']
        x = (hdr['XEnd'] - hdr['XStart']) / hdr['Resolution']
        t = hdr['volumes']
        return tuple(int(d) for d in [z, y, x, t])

    def set_data_shape(self, shape=None, zyx=None, t=None):
        ''' Set shape of data
        To conform with nibabel standards this implements shape.
        However, to fill the VtcHeader with sensible information use the zyxt parameter instead.

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        zyxt: 3x2 nested list [[1,2],[3,4],[5,6]]
           array storing borders of data
        t: int
           number of volumes

        '''
        if (shape is None) and (zyx is None) and (t is None):
            raise BvError('Shape, zyx, or t needs to be specified!')
        if shape is not None:
            # Use zyx and t parameters instead of shape. Dimensions will start from standard coordinates.
            if len(shape) != 4:
                raise BvError('Shape for VTC files must be 4 dimensional!')
            self._structarr['XEnd'] = 57 + (shape[2] * self._structarr['Resolution'])
            self._structarr['YEnd'] = 52 + (shape[1] * self._structarr['Resolution'])
            self._structarr['ZEnd'] = 59 + (shape[0] * self._structarr['Resolution'])
            self._structarr['volumes'] = shape[3]
            return
        self._structarr['XStart'] = zyx[2][0]
        self._structarr['XEnd'] = zyx[2][1]
        self._structarr['YStart'] = zyx[1][0]
        self._structarr['YEnd'] = zyx[1][1]
        self._structarr['ZStart'] = zyx[0][0]
        self._structarr['ZEnd'] = zyx[0][1]
        if t is not None:
            self._structarr['volumes'] = t

    def get_xflip(self):
        ''' Get xflip for data
        '''
        xflip = int(self._structarr['LRConvention'])
        if xflip == 1:
            return True
        elif xflip == 2:
            return False
        else:
            raise BvError('Left-right convention is unknown!')

    def set_xflip(self, xflip):
        ''' Set xflip for data
        '''
        if xflip == True:
            self._structarr['LRConvention'] = 1
        elif xflip == False:
            self._structarr['LRConvention'] = 2
        else:
            self._structarr['LRConvention'] = 0

    def update_template_dtype(self,binaryblock=None, item=None, value=None):
        if binaryblock is None:
            binaryblock = self.binaryblock

        # find length of fmr filename (start search after the first 2 bytes)
        # include the stop byte ('\x00') in the string
        fmrl = binaryblock.find('\x00', 2) - 1

        # find number of linked PRTs
        nPrt = int(np.fromstring(binaryblock[2+fmrl:2+fmrl+2],np.uint16))

        # find length of name(s) of linked PRT(s)
        prts = []
        point = 2 + fmrl + 3
        for prt in range(nPrt):
            prtl = binaryblock.find('\x00', point) - (point-2)
            prts.append(prtl)
            point += prtl

        # deep copy the template
        newTemplate = _make_vtc_header_dtd(fmrl, prts)

        # handle the items that should be changed
        if item is not None:
            newTemplate = [(x[0], x[1]) if x[0] != item else (item, 'S'+str(len(value)+1)) for x in newTemplate]
        
        dt = np.dtype(newTemplate)
        self.set_data_offset(dt.itemsize)
        self.template_dtype = dt

        return newTemplate

    @classmethod
    def default_structarr(klass, endianness=None):
        ''' Return header data for empty header with given endianness
        '''

        # create the template
        newTemplate = _make_vtc_header_dtd(1, ())

        dt = np.dtype(newTemplate)
        hdr = np.zeros((), dtype=dt)

        hdr['version'] = 3
        hdr['fmr'] = ''
        hdr['nPrt'] = 0
        hdr['prts']['prt0'] = ''
        hdr['currentPrt'] = 0
        hdr['datatype'] = 2
        hdr['volumes'] = 0
        hdr['Resolution'] = 3
        hdr['XStart'] = 57
        hdr['XEnd'] = 231
        hdr['YStart'] = 52
        hdr['YEnd'] = 172
        hdr['ZStart'] = 59
        hdr['ZEnd'] = 197
        hdr['LRConvention'] = 1
        hdr['RefSpace'] = 3
        hdr['TR'] = 2000

        return hdr

    @classmethod
    def _get_checks(klass):
        ''' Return sequence of check functions for this class '''
        return (klass._chk_fileversion,
                klass._chk_sizeof_hdr,
                klass._chk_datatype,
                )

    ''' Check functions in format expected by BatteryRunner class '''

    @classmethod
    def _chk_fileversion(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['version'] == 3:
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'only fileversion 3 is supported at the moment!'
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep

    @classmethod
    def _chk_sizeof_hdr(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr.template_dtype.itemsize == len(hdr.binaryblock):
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'size of binaryblock should be ' + str(hdr.template_dtype.itemsize)
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep

    @classmethod
    def _chk_datatype(klass, hdr, fix=False):
        rep = Report(HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dtype = klass._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 40
            rep.problem_msg = 'data code %d not recognized' % code
        else:
            if dtype.itemsize == 0:
                rep.problem_level = 40
                rep.problem_msg = 'data code %d not supported' % code
            else:
                return hdr, rep
        if fix:
            rep.fix_msg = 'not attempting fix'
        return hdr, rep

class VtcImage(BvFileImage):
    # Set the class of the corresponding header
    header_class = VtcHeader

    # Set the label ('image') and the extension ('.vtc') for a VTC file
    files_types = (('image', '.vtc'),)

load = VtcImage.load
save = VtcImage.instance_to_filename

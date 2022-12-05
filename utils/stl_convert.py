import sys
import os
import vtk
import itk
import numpy as np
import vtkmodules
# from vtkmodules.util import vtkImageExportToArray
# from vtk import vtkImageExport

ImageType = itk.Image[itk.SS, 3]
LabelType = itk.Image[itk.UC, 3]

# data_dir = '{}/image'.format(sys.path[0])
data_dir = '{}'.format('/zion/data/uronav_data')
save_dir = 'mriSeg_nii'

excluded_files = [] # excluded files

def convert():
    convert_num = 0
    for casename in os.listdir(data_dir):
        stl_fn = '{}/{}/segmentationrtss.uronav.stl'.format(data_dir, casename)
        mri_fn = '{}/{}/MRVol.mhd'.format(data_dir, casename)
        lb_fn = '{}/mr_label_{}.nii.gz'.format(save_dir, casename)
        if not (os.path.isfile(stl_fn) and os.path.isfile(mri_fn)):
            continue
        if casename in excluded_files:
            continue

        print(casename)

        polydata = loadStl(stl_fn)

        im_reader = itk.ImageFileReader[ImageType].New()
        im_reader.SetFileName(mri_fn)
        im_reader.UpdateOutputInformation()
        image = im_reader.GetOutput()
        size = image.GetLargestPossibleRegion().GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        whiteImage = vtk.vtkImageData()
        whiteImage.SetSpacing(spacing[0], spacing[1], spacing[2])
        whiteImage.SetDimensions(size[0], size[1], size[2])
        whiteImage.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
        whiteImage.SetOrigin(origin[0], origin[1], origin[2])
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        whiteImage.GetPointData().GetScalars().Fill(1)
        #n = whiteImage.GetNumberOfPoints()
        #for i in range(n):
        #    whiteImage.GetPointData().GetScalars().SetTuple1(i, 1)

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(origin[0], origin[1], origin[2])
        pol2stenc.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()

        v2a = vtkImageExportToArray.vtkImageExportToArray()
        v2a.SetInputData(imgstenc.GetOutput())
        array = v2a.GetArray()

        label = itk.GetImageFromArray(array)
        label.SetSpacing(spacing)
        label.SetOrigin(origin)
        print('mark')
        lb_writer = itk.ImageFileWriter[LabelType].New()
        lb_writer.SetFileName(lb_fn)
        lb_writer.SetInput(label)
        lb_writer.Update()

        convert_num += 1
    print('Converted total number: {0:d}'.format(convert_num))

def render():
    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    # Create a RenderWindowInteractor to permit manipulating the camera
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    stlFilename = '{}/image/Case0001/segmentationrtss.uronav.stl'.format(sys.path[0])
    polydata = loadStl(stlFilename)
    ren.AddActor(polyDataToActor(polydata))
    ren.SetBackground(0.1, 0.1, 0.1)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def loadStl(fname):
    """Load the given STL file, and return a vtkPolyData object for it."""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def polyDataToActor(polydata):
    """Wrap the provided vtkPolyData object in a mapper and an actor, returning
    the actor."""
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        #mapper.SetInput(reader.GetOutput())
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    #actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetColor(0.5, 0.5, 1.0)
    return actor

#render()

# convert()

def convertSTL2MHD(stl_path, img_path, output_path=None):
    print(stl_path)
    print(img_path)
    return 0


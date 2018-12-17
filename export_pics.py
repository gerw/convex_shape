#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

import sys
assert len(sys.argv) > 1
exstring = sys.argv[1]

if exstring == "example 6.1":
	name = 'basic'
	resolution = [2000, 2000]
	dim = 2
elif exstring == "example 6.2":
	name = 'nonsmooth'
	resolution = [2000, 1700]
	dim = 2
elif exstring == "example 6.3":
	name = 'convex_sphere'
	resolution = [2000, 1700]
	dim = 3
elif exstring == "example 6.3 without convexity":
	name = 'nonconvex_sphere'
	resolution = [2000, 1700]
	dim = 3
else:
	raise ValueError('Unknown value of "exstring": "%s"' % exstring)

# create a new 'PVD Reader'
solutionpvd = PVDReader(FileName='solutions/' + exstring + '/solution.pvd')
# Uninteresting stuff {{{
# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1159, 814]

# get color transfer function/color map for 'u'
uLUT = GetColorTransferFunction('u')
uLUT.LockDataRange = 0
uLUT.InterpretValuesAsCategories = 0
uLUT.ShowCategoricalColorsinDataRangeOnly = 0
uLUT.RescaleOnVisibilityChange = 0
uLUT.EnableOpacityMapping = 0
uLUT.RGBPoints = [-0.020682806941005036, 0.231373, 0.298039, 0.752941, 0.7718543097700776, 0.865003, 0.865003, 0.865003, 1.5643914264811603, 0.705882, 0.0156863, 0.14902]
uLUT.UseLogScale = 0
uLUT.ColorSpace = 'Diverging'
uLUT.UseBelowRangeColor = 0
uLUT.BelowRangeColor = [0.0, 0.0, 0.0]
uLUT.UseAboveRangeColor = 0
uLUT.AboveRangeColor = [1.0, 1.0, 1.0]
uLUT.NanColor = [1.0, 1.0, 0.0]
uLUT.Discretize = 1
uLUT.NumberOfTableValues = 256
uLUT.ScalarRangeInitialized = 1.0
uLUT.HSVWrap = 0
uLUT.VectorComponent = 0
uLUT.VectorMode = 'Magnitude'
uLUT.AllowDuplicateScalars = 1
uLUT.Annotations = []
uLUT.ActiveAnnotatedValues = []
uLUT.IndexedColors = []

# get opacity transfer function/opacity map for 'u'
uPWF = GetOpacityTransferFunction('u')
uPWF.Points = [-0.020682806941005036, 0.0, 0.5, 0.0, 1.5643914264811603, 1.0, 0.5, 0.0]
uPWF.AllowDuplicateScalars = 1
uPWF.ScalarRangeInitialized = 1

# show data in view
solutionpvdDisplay = Show(solutionpvd, renderView1)
# trace defaults for the display properties.
solutionpvdDisplay.Representation = 'Surface'
solutionpvdDisplay.AmbientColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.ColorArrayName = ['POINTS', 'u']
solutionpvdDisplay.DiffuseColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.LookupTable = uLUT
solutionpvdDisplay.MapScalars = 1
solutionpvdDisplay.InterpolateScalarsBeforeMapping = 1
solutionpvdDisplay.Opacity = 1.0
solutionpvdDisplay.PointSize = 2.0
solutionpvdDisplay.LineWidth = 1.0
solutionpvdDisplay.Interpolation = 'Gouraud'
solutionpvdDisplay.Specular = 0.0
solutionpvdDisplay.SpecularColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.SpecularPower = 100.0
solutionpvdDisplay.Ambient = 0.0
solutionpvdDisplay.Diffuse = 1.0
solutionpvdDisplay.EdgeColor = [0.0, 0.0, 0.5]
solutionpvdDisplay.BackfaceRepresentation = 'Follow Frontface'
solutionpvdDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.BackfaceOpacity = 1.0
solutionpvdDisplay.Position = [0.0, 0.0, 0.0]
solutionpvdDisplay.Scale = [1.0, 1.0, 1.0]
solutionpvdDisplay.Orientation = [0.0, 0.0, 0.0]
solutionpvdDisplay.Origin = [0.0, 0.0, 0.0]
solutionpvdDisplay.Pickable = 1
solutionpvdDisplay.Texture = None
solutionpvdDisplay.Triangulate = 0
solutionpvdDisplay.NonlinearSubdivisionLevel = 1
solutionpvdDisplay.UseDataPartitions = 0
solutionpvdDisplay.OSPRayUseScaleArray = 0
solutionpvdDisplay.OSPRayScaleArray = 'u'
solutionpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
solutionpvdDisplay.Orient = 0
solutionpvdDisplay.OrientationMode = 'Direction'
solutionpvdDisplay.SelectOrientationVectors = 'None'
solutionpvdDisplay.Scaling = 0
solutionpvdDisplay.ScaleMode = 'No Data Scaling Off'
solutionpvdDisplay.ScaleFactor = 0.2
solutionpvdDisplay.SelectScaleArray = 'u'
solutionpvdDisplay.GlyphType = 'Arrow'
solutionpvdDisplay.UseGlyphTable = 0
solutionpvdDisplay.GlyphTableIndexArray = 'u'
solutionpvdDisplay.UseCompositeGlyphTable = 0
solutionpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
solutionpvdDisplay.SelectionCellLabelBold = 0
solutionpvdDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
solutionpvdDisplay.SelectionCellLabelFontFamily = 'Arial'
solutionpvdDisplay.SelectionCellLabelFontSize = 18
solutionpvdDisplay.SelectionCellLabelItalic = 0
solutionpvdDisplay.SelectionCellLabelJustification = 'Left'
solutionpvdDisplay.SelectionCellLabelOpacity = 1.0
solutionpvdDisplay.SelectionCellLabelShadow = 0
solutionpvdDisplay.SelectionPointLabelBold = 0
solutionpvdDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
solutionpvdDisplay.SelectionPointLabelFontFamily = 'Arial'
solutionpvdDisplay.SelectionPointLabelFontSize = 18
solutionpvdDisplay.SelectionPointLabelItalic = 0
solutionpvdDisplay.SelectionPointLabelJustification = 'Left'
solutionpvdDisplay.SelectionPointLabelOpacity = 1.0
solutionpvdDisplay.SelectionPointLabelShadow = 0
solutionpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
solutionpvdDisplay.ScalarOpacityFunction = uPWF
solutionpvdDisplay.ScalarOpacityUnitDistance = 0.74264554600755
solutionpvdDisplay.SelectMapper = 'Projected tetra'
solutionpvdDisplay.SamplingDimensions = [128, 128, 128]
solutionpvdDisplay.UseFloatingPointFrameBuffer = 1

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
solutionpvdDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'Arrow' selected for 'GlyphType'
solutionpvdDisplay.GlyphType.TipResolution = 6
solutionpvdDisplay.GlyphType.TipRadius = 0.1
solutionpvdDisplay.GlyphType.TipLength = 0.35
solutionpvdDisplay.GlyphType.ShaftResolution = 6
solutionpvdDisplay.GlyphType.ShaftRadius = 0.03
solutionpvdDisplay.GlyphType.Invert = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
solutionpvdDisplay.DataAxesGrid.XTitle = 'X Axis'
solutionpvdDisplay.DataAxesGrid.YTitle = 'Y Axis'
solutionpvdDisplay.DataAxesGrid.ZTitle = 'Z Axis'
solutionpvdDisplay.DataAxesGrid.XTitleColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.XTitleFontFamily = 'Arial'
solutionpvdDisplay.DataAxesGrid.XTitleBold = 0
solutionpvdDisplay.DataAxesGrid.XTitleItalic = 0
solutionpvdDisplay.DataAxesGrid.XTitleFontSize = 12
solutionpvdDisplay.DataAxesGrid.XTitleShadow = 0
solutionpvdDisplay.DataAxesGrid.XTitleOpacity = 1.0
solutionpvdDisplay.DataAxesGrid.YTitleColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.YTitleFontFamily = 'Arial'
solutionpvdDisplay.DataAxesGrid.YTitleBold = 0
solutionpvdDisplay.DataAxesGrid.YTitleItalic = 0
solutionpvdDisplay.DataAxesGrid.YTitleFontSize = 12
solutionpvdDisplay.DataAxesGrid.YTitleShadow = 0
solutionpvdDisplay.DataAxesGrid.YTitleOpacity = 1.0
solutionpvdDisplay.DataAxesGrid.ZTitleColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.ZTitleFontFamily = 'Arial'
solutionpvdDisplay.DataAxesGrid.ZTitleBold = 0
solutionpvdDisplay.DataAxesGrid.ZTitleItalic = 0
solutionpvdDisplay.DataAxesGrid.ZTitleFontSize = 12
solutionpvdDisplay.DataAxesGrid.ZTitleShadow = 0
solutionpvdDisplay.DataAxesGrid.ZTitleOpacity = 1.0
solutionpvdDisplay.DataAxesGrid.FacesToRender = 63
solutionpvdDisplay.DataAxesGrid.CullBackface = 0
solutionpvdDisplay.DataAxesGrid.CullFrontface = 1
solutionpvdDisplay.DataAxesGrid.GridColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.ShowGrid = 0
solutionpvdDisplay.DataAxesGrid.ShowEdges = 1
solutionpvdDisplay.DataAxesGrid.ShowTicks = 1
solutionpvdDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
solutionpvdDisplay.DataAxesGrid.AxesToLabel = 63
solutionpvdDisplay.DataAxesGrid.XLabelColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.XLabelFontFamily = 'Arial'
solutionpvdDisplay.DataAxesGrid.XLabelBold = 0
solutionpvdDisplay.DataAxesGrid.XLabelItalic = 0
solutionpvdDisplay.DataAxesGrid.XLabelFontSize = 12
solutionpvdDisplay.DataAxesGrid.XLabelShadow = 0
solutionpvdDisplay.DataAxesGrid.XLabelOpacity = 1.0
solutionpvdDisplay.DataAxesGrid.YLabelColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.YLabelFontFamily = 'Arial'
solutionpvdDisplay.DataAxesGrid.YLabelBold = 0
solutionpvdDisplay.DataAxesGrid.YLabelItalic = 0
solutionpvdDisplay.DataAxesGrid.YLabelFontSize = 12
solutionpvdDisplay.DataAxesGrid.YLabelShadow = 0
solutionpvdDisplay.DataAxesGrid.YLabelOpacity = 1.0
solutionpvdDisplay.DataAxesGrid.ZLabelColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.DataAxesGrid.ZLabelFontFamily = 'Arial'
solutionpvdDisplay.DataAxesGrid.ZLabelBold = 0
solutionpvdDisplay.DataAxesGrid.ZLabelItalic = 0
solutionpvdDisplay.DataAxesGrid.ZLabelFontSize = 12
solutionpvdDisplay.DataAxesGrid.ZLabelShadow = 0
solutionpvdDisplay.DataAxesGrid.ZLabelOpacity = 1.0
solutionpvdDisplay.DataAxesGrid.XAxisNotation = 'Mixed'
solutionpvdDisplay.DataAxesGrid.XAxisPrecision = 2
solutionpvdDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
solutionpvdDisplay.DataAxesGrid.XAxisLabels = []
solutionpvdDisplay.DataAxesGrid.YAxisNotation = 'Mixed'
solutionpvdDisplay.DataAxesGrid.YAxisPrecision = 2
solutionpvdDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
solutionpvdDisplay.DataAxesGrid.YAxisLabels = []
solutionpvdDisplay.DataAxesGrid.ZAxisNotation = 'Mixed'
solutionpvdDisplay.DataAxesGrid.ZAxisPrecision = 2
solutionpvdDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
solutionpvdDisplay.DataAxesGrid.ZAxisLabels = []

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
solutionpvdDisplay.PolarAxes.Visibility = 0
solutionpvdDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
solutionpvdDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
solutionpvdDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
solutionpvdDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
solutionpvdDisplay.PolarAxes.EnableCustomRange = 0
solutionpvdDisplay.PolarAxes.CustomRange = [0.0, 1.0]
solutionpvdDisplay.PolarAxes.PolarAxisVisibility = 1
solutionpvdDisplay.PolarAxes.RadialAxesVisibility = 1
solutionpvdDisplay.PolarAxes.DrawRadialGridlines = 1
solutionpvdDisplay.PolarAxes.PolarArcsVisibility = 1
solutionpvdDisplay.PolarAxes.DrawPolarArcsGridlines = 1
solutionpvdDisplay.PolarAxes.NumberOfRadialAxes = 0
solutionpvdDisplay.PolarAxes.AutoSubdividePolarAxis = 1
solutionpvdDisplay.PolarAxes.NumberOfPolarAxis = 0
solutionpvdDisplay.PolarAxes.MinimumRadius = 0.0
solutionpvdDisplay.PolarAxes.MinimumAngle = 0.0
solutionpvdDisplay.PolarAxes.MaximumAngle = 90.0
solutionpvdDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
solutionpvdDisplay.PolarAxes.Ratio = 1.0
solutionpvdDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.PolarAxisTitleVisibility = 1
solutionpvdDisplay.PolarAxes.PolarAxisTitle = 'Radial Distance'
solutionpvdDisplay.PolarAxes.PolarAxisTitleLocation = 'Bottom'
solutionpvdDisplay.PolarAxes.PolarLabelVisibility = 1
solutionpvdDisplay.PolarAxes.PolarLabelFormat = '%-#6.3g'
solutionpvdDisplay.PolarAxes.PolarLabelExponentLocation = 'Labels'
solutionpvdDisplay.PolarAxes.RadialLabelVisibility = 1
solutionpvdDisplay.PolarAxes.RadialLabelFormat = '%-#3.1f'
solutionpvdDisplay.PolarAxes.RadialLabelLocation = 'Bottom'
solutionpvdDisplay.PolarAxes.RadialUnitsVisibility = 1
solutionpvdDisplay.PolarAxes.ScreenSize = 10.0
solutionpvdDisplay.PolarAxes.PolarAxisTitleColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
solutionpvdDisplay.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
solutionpvdDisplay.PolarAxes.PolarAxisTitleBold = 0
solutionpvdDisplay.PolarAxes.PolarAxisTitleItalic = 0
solutionpvdDisplay.PolarAxes.PolarAxisTitleShadow = 0
solutionpvdDisplay.PolarAxes.PolarAxisTitleFontSize = 12
solutionpvdDisplay.PolarAxes.PolarAxisLabelColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
solutionpvdDisplay.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
solutionpvdDisplay.PolarAxes.PolarAxisLabelBold = 0
solutionpvdDisplay.PolarAxes.PolarAxisLabelItalic = 0
solutionpvdDisplay.PolarAxes.PolarAxisLabelShadow = 0
solutionpvdDisplay.PolarAxes.PolarAxisLabelFontSize = 12
solutionpvdDisplay.PolarAxes.LastRadialAxisTextColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
solutionpvdDisplay.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
solutionpvdDisplay.PolarAxes.LastRadialAxisTextBold = 0
solutionpvdDisplay.PolarAxes.LastRadialAxisTextItalic = 0
solutionpvdDisplay.PolarAxes.LastRadialAxisTextShadow = 0
solutionpvdDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextColor = [1.0, 1.0, 1.0]
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
solutionpvdDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
solutionpvdDisplay.PolarAxes.EnableDistanceLOD = 1
solutionpvdDisplay.PolarAxes.DistanceLODThreshold = 0.7
solutionpvdDisplay.PolarAxes.EnableViewAngleLOD = 1
solutionpvdDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
solutionpvdDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
solutionpvdDisplay.PolarAxes.PolarTicksVisibility = 1
solutionpvdDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
solutionpvdDisplay.PolarAxes.TickLocation = 'Both'
solutionpvdDisplay.PolarAxes.AxisTickVisibility = 1
solutionpvdDisplay.PolarAxes.AxisMinorTickVisibility = 0
solutionpvdDisplay.PolarAxes.ArcTickVisibility = 1
solutionpvdDisplay.PolarAxes.ArcMinorTickVisibility = 0
solutionpvdDisplay.PolarAxes.DeltaAngleMajor = 10.0
solutionpvdDisplay.PolarAxes.DeltaAngleMinor = 5.0
solutionpvdDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
solutionpvdDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
solutionpvdDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
solutionpvdDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
solutionpvdDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
solutionpvdDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
solutionpvdDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
solutionpvdDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
solutionpvdDisplay.PolarAxes.ArcMajorTickSize = 0.0
solutionpvdDisplay.PolarAxes.ArcTickRatioSize = 0.3
solutionpvdDisplay.PolarAxes.ArcMajorTickThickness = 1.0
solutionpvdDisplay.PolarAxes.ArcTickRatioThickness = 0.5
solutionpvdDisplay.PolarAxes.Use2DMode = 0
solutionpvdDisplay.PolarAxes.UseLogAxis = 0

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.CameraPosition = [0.0, -5.551115123125783e-17, 10000.0]

# show color bar/color legend
solutionpvdDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# change representation type
if dim == 2:
	solutionpvdDisplay.SetRepresentationType('Wireframe')
else:
	solutionpvdDisplay.SetRepresentationType('Surface With Edges')

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(uLUT, renderView1)

# change solid color
solutionpvdDisplay.AmbientColor = [0.0, 0.0, 0.0]

# current camera placement for renderView1
if dim == 2:
	renderView1.InteractionMode = '2D'
	renderView1.CameraPosition = [0.0, -5.551115123125783e-17, 10000.0]
	renderView1.CameraFocalPoint = [0.0, -5.551115123125783e-17, 0.0]
	renderView1.CameraParallelScale = 1.05
else:
	renderView1.InteractionMode = '3D'
	renderView1.CameraPosition = [0.65*f for f in [3.81089138609945, 0.542285801481981, -5.43395791394274]]
	renderView1.CameraFocalPoint = [0,0,0]
	renderView1.CameraViewUp = [0.543364516088308,-0.782934240238395, 0.302933950092172]
	# renderView1.CameraParallelScale = 2.0
# }}}

ColorBy(solutionpvdDisplay, None)
solutionpvdDisplay.AmbientColor = [0.,0.,0.]

# hide color bar/color legend
solutionpvdDisplay.SetScalarBarVisibility(renderView1, False)
HideScalarBarIfNotNeeded(uLUT, renderView1)

# Properties modified on solutionpvdDisplay
solutionpvdDisplay.LineWidth = 3.0

f = open('solutions/' + exstring + '/views.py')
views = eval(f.read())


animationScene1 = GetAnimationScene()

for i in range(len(views)):
	view = views[i]

	animationScene1.GoToFirst()
	for j in range(view - 1):
		animationScene1.GoToNext()

	# save screenshot
	SaveScreenshot('pics/' + name + '_%02d.png' % i, renderView1,
			ImageResolution=resolution,
			FontScaling='Scale fonts proportionally',
			OverrideColorPalette='PrintBackground',
			StereoMode='No change',
			TransparentBackground=0,
			ImageQuality=100)
	# ExportView('/home/wachsmut/work/publications/current/convex_shape_optimization/pics/test_%02d.pdf' % i, view=renderView1, Rasterize3Dgeometry=0)

# vim: fdm=marker

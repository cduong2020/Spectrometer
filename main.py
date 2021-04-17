from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import sys
import os
import cv2
import time

# Variables
save_seq = 0                                                # images identification
calibrationType= ""                                         # variable to hold which calibration type the user is currently choosing to properly save the images
simpleCalibrationImages = []                                # store images used for simple calibration
advancedCalibrationImage = []                               # store image used for advance calibration
color = []                                                  # store light colors used in calibration
givenWavelenghts = []                                       # store given wavelengths used in calibration
calculatedWaveleghts = ["wavelenght 1","wavelenght 2","wavelenght 3","wavelenght 4"]            # wavelenghts caculated to be on the unknwon spectra images
unknownSpectraImage = []                                    # store image used for analysis

class confirmPopup(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.title = 'Confirm'
        self.setWindowTitle(self.title)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.confirmlabel = QLabel("Ready to analyze spectra!")
        self.OK = QPushButton("OK")
        self.OK.clicked.connect(self.close)

        self.layout.addWidget(self.confirmlabel)
        self.layout.addWidget(self.OK)

class calibratePopup(QWidget):                              # Widget for calibrate
    def __init__(self):
        QWidget.__init__(self)

        self.title = 'Calibration'
        self.setWindowTitle(self.title)
        self.qlayout = QGridLayout()
        self.setLayout(self.qlayout)

        self.confirmspectra = None

        self.statusbar = QStatusBar()                       #status bar for info
        self.statusbar.setStyleSheet("background : white;")
        self.setStatusBar = (self.statusbar)

        self.labelcolor = QLabel("Color:")
        self.editcolor = QLineEdit()
        #self.
        self.labelwavelength = QLabel("Wavelength:")
        self.editwavelength = QLineEdit()

        self.NextPicture = QPushButton("Next Picture")      #closes popup - take another picture on main widget
        self.NextPicture.setStatusTip("Closes popup - click Calibrate on Main Window to take another picture!")
        self.NextPicture.clicked.connect(self.close)
        #nextpicture click = info from color and wavelengths saved to color and wavelength variables
        #calibrate click = pic saved to simple or advanced folders

        self.Done = QPushButton("Done", self)               #
        self.Done.clicked.connect(self.confirm)
        self.Done.clicked.connect(self.close)
        self.Done.setStatusTip("Placeholder")

        self.qlayout.addWidget(self.labelcolor)
        self.qlayout.addWidget(self.editcolor)
        self.qlayout.addWidget(self.labelwavelength)
        self.qlayout.addWidget(self.NextPicture)
        self.qlayout.addWidget(self.Done)
        self.qlayout.addWidget(self.statusbar)

    @pyqtSlot()
    def confirm(self):
        self.confirmspectra = confirmPopup()
        self.confirmspectra.setGeometry(1000, 700, 300, 300)
        self.confirmspectra.show()

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.title = 'Spectrometer'
        self.left = 1500
        self.top = 700
        self.width = 800
        self.height = 700
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon('logo.png'))
        self.setGeometry(self.left,self.top,self.width,self.height)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

       #CAPTURE TAB - has capture button and camera view
        self.tab1 = QWidget()
        self.tab1.layout = QVBoxLayout()                            # creating layout: vertically add widgets to this tab

        self.groupboxWebcam = QGroupBox("Webcam View")              # creating a groupbox w/title to put Camera Feed
        self.vbox1 = QVBoxLayout()                                  # layout of groupbox (verticall)
        self.groupboxWebcam.setLayout(self.vbox1)                   # add vbox layout to groupbox

        self.available_cameras = QCameraInfo.availableCameras()     # getting available cameras
        if not self.available_cameras:                              # if no camera found
            # exit the code
            #sys.exit()
            pass

        self.save_path = ""                                         # path to save

        self.camera = QCamera()                                     # creating camera object
        self.viewfinder = QCameraViewfinder()                       # creating a QCameraViewfinder object to vizualize camera
        self.camera.setViewfinder(self.viewfinder)                  # setting view finder to the camera
        self.camera.setCaptureMode(QCamera.CaptureStillImage)       # setting capture mode to the camera
        self.camera.error.connect(lambda: self.alert(self.camera.errorString())) # if any error occur show the alert
        self.camera.start()                                         # start the camera
        self.viewfinder.show()

        self.capture = QCameraImageCapture(self.camera)             # creating a QCameraImageCapture object
        self.capture.error.connect(lambda error_msg, error,         # showing alert if error occur
                                   msg: self.alert(msg))
        #self.save_seq = 0                                          # inital save sequence
        self.vbox1.addWidget(self.viewfinder)                       # adding Camera view widget to vbbox inside groupbox

        self.popup = None                                           # object for the popup calibration widget

        self.hbox1 = QWidget()                                      # Horizontal layout box to add buttons
        self.hbox1.layout = QHBoxLayout()
        self.calibrateButton = QPushButton("Calibrate")
        self.calibrateButton.clicked.connect(self.capture_image)    # calibrate button takes a picture and opens popup
        self.calibrateButton.clicked.connect(self.calibrate)
        self.analyzeButton = QPushButton("Analyze")
        self.analyzeButton.clicked.connect(self.capture_image)      # analyze button takes a picture and goes to analysis tab
        self.analyzeButton.clicked.connect(self.analysis)
        self.change_folder_action = QPushButton("Save Location")    # save folder button to choose where to save pictures
        self.change_folder_action.clicked.connect(self.change_folder)

        self.groupboxRadioButtons = QGroupBox("Calibration Type")   # creating a groupbox w/title to put Radio buttons
        self.vbox2 = QVBoxLayout()
        self.groupboxRadioButtons.setLayout(self.vbox2)
        self.simple_radio_button = QRadioButton(self)
        self.simple_radio_button.setText("Simple")
        self.simple_radio_button.clicked.connect(lambda:self.check(self.simple_radio_button))
        self.advanced_radio_button = QRadioButton(self)
        self.advanced_radio_button.setText("Advanced")
        self.advanced_radio_button.clicked.connect(lambda:self.check(self.advanced_radio_button ))
        self.vbox2.addWidget(self.simple_radio_button)              # adding simple radio button widget to vbox2 inside groupbox
        self.vbox2.addWidget(self.advanced_radio_button)            # adding advanced radio button widget to vbox2 inside groupbox
        self.vbox2.addWidget(self.calibrateButton)                  # add buttons to hbox

        self.hbox1.layout.addWidget(self.groupboxRadioButtons)      # add groupbox radio buttons to hbox
        self.hbox1.layout.addWidget(self.analyzeButton)
        self.hbox1.layout.addWidget(self.change_folder_action)
        self.hbox1.setLayout(self.hbox1.layout)                     # hbox end - set horizonta layout

        self.tab1.layout.addWidget(self.groupboxWebcam)             # adding groupbox w/ Camera View to tab1
        self.tab1.layout.addWidget(self.hbox1)                      # adding buttons box to tab1
        self.tab1.setLayout(self.tab1.layout)                       # setting vertical layout to tab1

        # ANALYSIS TAB - wavelenght list, captured image, plots
        self.tab2 = QWidget()
        self.tab2.layout = QHBoxLayout()                            # Horizontal layout

        self.hbox2 =  QWidget()                                     # Image + plot in vertical box
        self.hbox2.layout = QVBoxLayout()

        self.groupboxTab2_1 = QGroupBox( "Spectrum Image")
        self.vbox3 = QVBoxLayout()
        self.groupboxTab2_1.setLayout(self.vbox3)
        self.capturedImage = QLabel("Captured Image")               # captured image from camera
        self.pixmap2 = QPixmap('spectrum3.png')
        self.capturedImage.setPixmap(self.pixmap2)
        self.vbox3.addWidget(self.capturedImage)#

        self.groupboxTab2_2 = QGroupBox( "Intensity vs Wavelenght Plot")
        self.vbox4 = QVBoxLayout()
        self.groupboxTab2_2.setLayout(self.vbox4)
        self.plotImage = QLabel("Plot")                             # plot
        self.pixmap1 = QPixmap('histogram.png')
        self.plotImage.setPixmap(self.pixmap1)
        self.vbox4.addWidget(self.plotImage)#

        self.hbox2.layout.addWidget(self.groupboxTab2_1)
        self.hbox2.layout.addWidget(self.groupboxTab2_2)
        self.hbox2.setLayout(self.hbox2.layout)

        self.hbox3 =  QWidget()                                     # List + buttons in vertical plot
        self.hbox3.layout = QVBoxLayout()

        self.groupboxTab2_3 = QGroupBox( "Wavelenght")
        self.vbox5 = QVBoxLayout()
        self.groupboxTab2_3.setLayout(self.vbox5)
        self.wavelenghtList = QListWidget()                         # list of wavelenghts
        self.vbox5.addWidget(self.wavelenghtList)


        self.reportButton = QPushButton("Generate Report")
        self.reportButton.clicked.connect(self.generateReport)
        self.resetButton = QPushButton("Take Another Picture")
        self.resetButton.clicked.connect(self.reset)

        self.hbox3.layout.addWidget(self.groupboxTab2_3)
        self.hbox3.layout.addWidget(self.reportButton)
        self.hbox3.layout.addWidget(self.resetButton)
        self.hbox3.setLayout(self.hbox3.layout)

        self.tab2.layout.addWidget(self.hbox2)                      # add hbox2(Image+plot) to tab2
        self.tab2.layout.addWidget(self.hbox3)                      # add hbox3(list+buttons)to tab2
        self.tab2.setLayout(self.tab2.layout)                       # set horizontal layout to tab2

        # OVERALL TABS
        self.tabwidget = QTabWidget()
        self.tabwidget.addTab(self.tab1, "Capture",)
        self.tabwidget.addTab(self.tab2, "Analysis")
        self.layout.addWidget(self.tabwidget, 0, 0)

# SIGNALS

    @pyqtSlot()
    def capture_image(self):                                       # capture image
        print('IMAGE CAPTURED')
        global save_seq
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")             # time stamp
        self.capture.capture(os.path.join(self.save_path,          # capture the image and save it on the save path
                                            "%d.jpg" %(
                    save_seq,
                    #timestamp #"%04d-%s.jpg"
                )))
        save_seq = save_seq + 1                                     # increment the sequence
        #Put images in the correct list
        if (calibrationType == "Simple"):
            simpleCalibrationImages.append(cv2.imread('%d.jpg' % (save_seq-1)))
        elif (calibrationType == "Advanced"):
            advancedCalibrationImages.append(cv2.imread('%d.jpg' % (save_seq-1)))
        else:
            unknownSpectraImage.append(cv2.imread('%d.jpg' % (save_seq-1)))

        # TEST to check if its saving images to list
        if (len(simpleCalibrationImages)>0):
            print("Simple calibration image is saved")
        if (len(advancedCalibrationImage)>0):
            print("Advanced calibration image is saved")

    @pyqtSlot()
    def analysis(self):
        # put calculated wavelenghts into the List widget
        for i in calculatedWaveleghts:
            self.wavelenghtList.addItem(i)
        # image of plot to its space
        # image of unknown spectrum to space
        self.tabwidget.setCurrentIndex(1)                          # goes to analysis tab

    @pyqtSlot()
    def check(self,rb):                                            # check which calibration method was chosen
        if rb.text() == "Simple":
            calibrationType = rb.text()
        else:
            calibrationType = rb.text()
        print(calibrationType)

    @pyqtSlot()
    def calibrate(self):                                            # when calibrate is pressed popup shows up
        self.popup = calibratePopup()
        self.popup.setGeometry(900, 700, 500, 500)
        self.popup.show()

    @pyqtSlot()
    def generateReport(self):
        print('Report')
        # generate excelpdf report with the data and opne a popup to save the file

    @pyqtSlot()
    def reset(self):
        print('Take Another Picture')
        # clean calculated wavelenghts data, unknown spectra image, decrease save sequence and clean plot
        self.tabwidget.setCurrentIndex(0)                          # goes to analysis tab

    @pyqtSlot()
    def change_folder(self):

        # open the dialog to select path
        path = QFileDialog.getExistingDirectory(self, "Picture Location", "")

        # if path is selected
        if path:

            # update the path
            self.save_path = path

            # update the sequence
            self.save_seq = 0


app = QApplication(sys.argv)
screen = Window()
screen.show()
sys.exit(app.exec_())

#NEXT STEPS
# IMPORTANT
    # Camera DONE
        # Figure it out camera Feed (QMultimedia)- OK
        # Figure it out capturing image - OK
        # Figure it out saving on the right place -> should be on the same folder as everything  - OK

    # Methods
        # calibrateSimple()*****
            # notebook - OK
            # Turn it to method - OK
            # receives the 3 laser pictures with respective wavelenghts
            # returns parameters a and b of wavelenght = a * pixelX + b
        # calibrateAdvanced()*****
            # notebook - OK
            # Turn it to method - OK
            # receives the 1 laser flourescent with respective wavelenghts, a and b parameters
            # returns parameters of non-linear
        # analyze()*****
            # notebook - OK
            # receives parameters,spectrum picture
            # returns wavelenght list and plot (Call intensityPlot()???)
        # resize()***
            # Image resizing -> this is important to display picture on the app ( PIL library,cobtours+crop image) (dont change original) make a method
        # getReport()****
            # pdf/excel report of the data with date and time
            # receives the laser colors and wavelenghts, parameter a and b, spectrum image, wavelenght list and plot

    # Interface Design
        # Capture TAB
            # Add a button calibrate - OK
                # if clicked show a pop up window (Calibrate Settings) - OK
                    # text with instructions on how to calibrate
                    # label color followed by a user input line - OK
                    # label wavelenght followed by insert multiples inputs line
                    # button Next Image - OK
                        # -> when clicked goes back to capture tab and closes the popup window
                    # button Done - OK
                        # -> when clicked set varible calibrationtype="", perfom the calculations of the parameters from calibration, popups a message and goes back to capture tab, closes calibration popup window
                    # status bar to help user navigate
            # analysis button functionality
                # calculate wavelenghts, and create plot
                # Iterate through the calculated wavelenghts and add them to list widget - OK
                # put image plot on its place
                # resize spectra image and put on its place

        # Analysis TAB
            # add a generate report button
                # if clicked shows pop up window() to choose the save location
            # add take another picture button
                # cleans the analysis data(wavelenghts,plot and spectra)
                # takes you to capture tab

        # About TAB
            # add image logo
            # add text
            # add button Start -> takes to capture tab

# DETAILS
    # set title to sections (using QGroupBox layouts) tab1 - OK, tab2 - OK
    # Organize tab2 layout better (widgets size and placement)  - OK
    # disable button DONE on the calibration tab until you have at least 3 pictures taken
    

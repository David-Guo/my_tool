QT += \
    widgets \
    gui

HEADERS += \
    ScrollAreaNoWheel.h \
    PixmapWidget.h \
    MainWindow.h \
    ImgAnnotation.h \
    defines.h

SOURCES += \
    ScrollAreaNoWheel.cpp \
    PixmapWidget.cpp \
    MainWindow.cpp \
    main.cpp \
    ImgAnnotation.cpp

FORMS += \
    MainWindow.ui

# 终端执行 sudo apt-get install qt5-default qtcreator 安装 qt5
LIBS += -L/usr/lib
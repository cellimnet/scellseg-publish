try:
    from scellseg.guis import scellsegGui
    GUI_ENABLED = True 
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise

def main():
    if not GUI_ENABLED:
        print('ERROR: %s'%GUI_ERROR)
        if GUI_IMPORT:
            print('GUI FAILED: GUI dependencies may not be installed, to install, run')
            print('pip install scellseg')
    else:
        scellsegGui.start_gui()


if __name__ == '__main__':
    main()

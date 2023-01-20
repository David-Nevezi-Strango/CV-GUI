import autoGUI
import argparse
#import fingers
#import handsign

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CV-GUI CLI Application')
    parser.add_argument('-s', '--start_photo', metavar='FILE', type=str, default='start.png',
                        help='path to start icon screen capture')
    parser.add_argument('-v', '--version', action='version', version='1.0')

    args = parser.parse_args()
    start_photo = args.start_photo
    print("\n")
    #fingers.run()
    autoGUI.run(start_photo)

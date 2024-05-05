from unittest import TestCase
from src.project_gpu import main
from    pathlib import Path

class Test(TestCase):
    def test_integration(self):
        # mock the arguments
        class Args:
            inputImage = './data/input/test.png'
            outputImage = './data/output/test_bw.jpg'
            tb = 32
            bw = False
            gauss = False
            sobel = False
            non_max_suppressed = False
            threshold = False

            hysteresis = False


        args = Args()
        # make the output directory
        Path('./data/output').mkdir(parents=True, exist_ok=True)
        args.bw = True
        main(args)

        args = Args()
        args.gauss = True
        args.outputImage = './data/output/test_gauss.jpg'
        main(args)

        args = Args()
        args.sobel = True
        args.outputImage = './data/output/test_sobel.jpg'
        main(args)

        args = Args()
        args.non_max_suppressed = True
        args.outputImage = './data/output/test_non_max_suppressed.jpg'
        main(args)

        args = Args()
        args.threshold = True
        args.outputImage = './data/output/test_threshold.jpg'
        main(args)


        args = Args()
        args.hysteresis = True
        args.outputImage = './data/output/test_hysteresis.jpg'
        main(args)



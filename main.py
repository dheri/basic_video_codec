import ex1
import ex2

if __name__ == "__main__":
    input_file = 'data/foreman_cif.yuv'
    width, height = 352, 288

    ex1.main(input_file, width, height)
    ex2.main(input_file, width, height)


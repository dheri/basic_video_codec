import ex1
import ex2

if __name__ == "__main__":
    input_file = 'data/foreman_cif-1.yuv'

    output_file = 'data/foreman_cif_noizy.yuv'
    width, height = 352, 288

    # ex1.main(input_file, 'data/foreman_cif_noizy.yuv', width, height)
    ex2.main(input_file, 'data/foreman_cif_y.y', width, height)


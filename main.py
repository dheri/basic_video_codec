import ex1

if __name__ == "__main__":
    input_file = 'data/foreman_cif-1.yuv'

    output_file = 'data/foreman_cif'
    width, height = 352, 288
    ex1.main(input_file, output_file, width, height)


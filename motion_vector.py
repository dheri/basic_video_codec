def parse_mv(mv_str: str):
    mv_field = {}
    mv_blocks = mv_str.strip().split('|')
    for b in mv_blocks[:-1]:  # ignore last element which will be empty
        kv_pairs = b.split(':')
        cords_txt = kv_pairs[0].split(',')
        mv_txt = kv_pairs[1].split(',')
        cords = (int(cords_txt[0]), int(cords_txt[1]))
        mv = [int(mv_txt[0]), int(mv_txt[1])]
        mv_field[cords] = mv
    return mv_field

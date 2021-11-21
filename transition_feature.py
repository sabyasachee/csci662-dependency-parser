# author - Sabyasachee

def create_transition_feature(stack, buffer):
    '''
    create a 48-dimensional feature list for given stack and buffer configuration.
    both stack and buffer are Stack data structures.
    
    s_i is the i-th tree node from the top in the stack, counting from 1.
    b_i is the i-th tree node from the top in the buffer, counting from 1.
    lc_i(x) is the i-th child tree node from the left of tree node x, counting from 1.
    rc_i(x) is the i-th child tree node from the right of tree node x, counting from 1.
    x.w is the word of tree node x.
    x.t is the pos tag of tree node x.
    x.l is the dependency arc between tree node x and the parent tree node of tree node x.

    feature[0]  = s_1.w                     feature[18] = s_1.t
    feature[1]  = s_2.w                     feature[19] = s_2.t
    feature[2]  = s_3.w                     feature[20] = s_3.t
    feature[3]  = b_1.w                     feature[21] = b_1.t
    feature[4]  = b_2.w                     feature[22] = b_2.t
    feature[5]  = b_3.w                     feature[23] = b_3.t

    feature[6]  = lc_1(s_1).w               feature[24] = lc_1(s_1).t               feature[36] = lc_1(s_1).l
    feature[7]  = rc_1(s_1).w               feature[25] = rc_1(s_1).t               feature[37] = rc_1(s_1).l
    feature[8]  = lc_2(s_1).w               feature[26] = lc_2(s_1).t               feature[38] = lc_2(s_1).l
    feature[9]  = rc_2(s_1).w               feature[27] = rc_2(s_1).t               feature[39] = rc_2(s_1).l
    feature[10] = lc_1(s_2).w               feature[28] = lc_1(s_2).t               feature[40] = lc_1(s_2).l
    feature[11] = rc_1(s_2).w               feature[29] = rc_1(s_2).t               feature[41] = rc_1(s_2).l
    feature[12] = lc_2(s_2).w               feature[30] = lc_2(s_2).t               feature[42] = lc_2(s_2).l
    feature[13] = rc_2(s_2).w               feature[31] = rc_2(s_2).t               feature[43] = rc_2(s_2).l

    feature[14] = lc_1(lc_1(s_1)).w         feature[32] = lc_1(lc_1(s_1)).t         feature[44] = lc_1(lc_1(s_1)).l
    feature[15] = rc_1(rc_1(s_1)).w         feature[33] = rc_1(rc_1(s_1)).t         feature[45] = rc_1(rc_1(s_1)).l
    feature[16] = lc_1(lc_1(s_2)).w         feature[34] = lc_1(lc_1(s_2)).t         feature[46] = lc_1(lc_1(s_2)).l
    feature[17] = rc_1(rc_1(s_2)).w         feature[35] = rc_1(rc_1(s_2)).t         feature[47] = rc_1(rc_1(s_2)).l

    if a tree node x is not present, then x.w = NULLWORD, x.t = NULLPOS, x.l = NULLARC

    s_i is accessed as stack[i][0] because stack items are tuples, and the unparsed tree nodes are always present at the first position of the tuple
    similarly for buffer, b_i is accessed as buffer[i][0]
    '''
    word_feature = ["NULLWORD" for _ in range(18)]
    pos_feature = ["NULLPOS" for _ in range(18)]
    arc_feature = ["NULLARC" for _ in range(12)]

    if len(stack):
        word_feature[0] = stack[1][0].word
        pos_feature[0] = stack[1][0].coarse_POS

    if len(stack) > 1:
        word_feature[1] = stack[2][0].word
        pos_feature[1] = stack[2][0].coarse_POS

    if len(stack) > 2:
        word_feature[2] = stack[3][0].word
        pos_feature[2] = stack[3][0].coarse_POS

    if len(buffer):
        word_feature[3] = buffer[1][0].word
        pos_feature[3] = buffer[1][0].coarse_POS

    if len(buffer) > 1:
        word_feature[4] = buffer[2][0].word
        pos_feature[4] = buffer[2][0].coarse_POS

    if len(buffer) > 2:
        word_feature[5] = buffer[3][0].word
        pos_feature[5] = buffer[3][0].coarse_POS
    
    if len(stack) and len(stack[1][0].children):
        leftmost_child = stack[1][0].children[0]
        rightmost_child = stack[1][0].children[-1]
        
        word_feature[6] = leftmost_child.word
        pos_feature[6] = leftmost_child.coarse_POS
        arc_feature[0] = leftmost_child.arc_label
        
        word_feature[7] = rightmost_child.word
        pos_feature[7] = rightmost_child.coarse_POS
        arc_feature[1] = rightmost_child.arc_label

        if len(stack[1][0].children) > 1:
            second_leftmost_child = stack[1][0].children[1]
            second_rightmost_child = stack[1][0].children[-2]

            word_feature[8] = second_leftmost_child.word
            pos_feature[8] = second_leftmost_child.coarse_POS
            arc_feature[2] = second_leftmost_child.arc_label

            word_feature[9] = second_rightmost_child.word
            pos_feature[9] = second_rightmost_child.coarse_POS
            arc_feature[3] = second_rightmost_child.arc_label

        if len(leftmost_child.children):
            leftmost_leftmost_child = leftmost_child.children[0]
            word_feature[14] = leftmost_leftmost_child.word
            pos_feature[14] = leftmost_leftmost_child.coarse_POS
            arc_feature[8] = leftmost_leftmost_child.arc_label

        if len(rightmost_child.children):
            rightmost_rightmost_child = rightmost_child.children[-1]
            word_feature[15] = rightmost_rightmost_child.word
            pos_feature[15] = rightmost_rightmost_child.coarse_POS
            arc_feature[9] = rightmost_rightmost_child.arc_label

    if len(stack) > 1 and len(stack[2][0].children):
        leftmost_child = stack[2][0].children[0]
        rightmost_child = stack[2][0].children[-1]

        word_feature[10] = leftmost_child.word
        pos_feature[10] = leftmost_child.coarse_POS
        arc_feature[4] = leftmost_child.arc_label

        word_feature[11] = rightmost_child.word
        pos_feature[11] = rightmost_child.coarse_POS
        arc_feature[5] = rightmost_child.arc_label

        if len(stack[2][0].children) > 1:
            second_leftmost_child = stack[2][0].children[1]
            second_rightmost_child = stack[2][0].children[-1]

            word_feature[12] = second_leftmost_child.word
            pos_feature[12] = second_leftmost_child.coarse_POS
            arc_feature[6] = second_leftmost_child.arc_label

            word_feature[13] = second_rightmost_child.word
            pos_feature[13] = second_rightmost_child.coarse_POS
            arc_feature[7] = second_rightmost_child.arc_label

        if len(leftmost_child.children):
            leftmost_leftmost_child = leftmost_child.children[0]
            word_feature[16] = leftmost_leftmost_child.word
            pos_feature[16] = leftmost_leftmost_child.coarse_POS
            arc_feature[10] = leftmost_leftmost_child.arc_label

        if len(rightmost_child.children):
            rightmost_rightmost_child = rightmost_child.children[-1]
            word_feature[17] = rightmost_rightmost_child.word
            pos_feature[17] = rightmost_rightmost_child.coarse_POS
            arc_feature[11] = rightmost_rightmost_child.arc_label

    feature = word_feature + pos_feature + arc_feature
    return feature
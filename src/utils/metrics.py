import numpy as np



class Metrics(object):

    def __init__(self, score_file, num_cands):
        super(Metrics, self).__init__()
        self.score_file = score_file
        self.num_cands = num_cands

    def __read_score_file(self):
        sessions = []
        one_sess = []
        with open(self.score_file, 'r') as infile:
            i = 0
            for line in infile.readlines():
                i += 1
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), int(tokens[1])))
                if i % self.num_cands == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                    else:
                        print('this session has no positive example')
                    one_sess = []
        return sessions

    def __recall_at_position_k(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)

    # cal mrr
    def reciprocal_rank(self,sort_data):
        rr = 0.0
        for i,item in enumerate(sort_data):
            score,label = item
            if label > 0:
                rr = 1.0/(i+1)
                break
        assert rr != 0.0
        return rr

    def evaluation_one_session(self, data):
        '''
        :param data: one conversion session(actually it means one turn), which layout is [(score1, label1), (score2, label2), ..., (score20, label20)].
        :return: all kinds of metrics used in paper.
        '''
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        
        r_1 = self.__recall_at_position_k(sort_data, 1)
        r_2 = self.__recall_at_position_k(sort_data, 2)
        r_5 = self.__recall_at_position_k(sort_data, 5)
        rr = self.reciprocal_rank(sort_data)
        return r_1, r_2, r_5, rr

    def evaluate_all_metrics(self):
        # 如果直接做20分类，那么R1@20就是acc
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0
        sum_rr = 0
        sessions = self.__read_score_file() # 20个example 为一个session
        total_s = len(sessions)
        print('total_s : {}'.format(total_s))
        for session in sessions:
            r_1, r_2, r_5, rr = self.evaluation_one_session(session)
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5
            sum_rr += rr 
        print('mrr : {}'.format(sum_rr/(total_s)))

        return (sum_r_1/total_s,
                sum_r_2/total_s,
                sum_r_5/total_s)



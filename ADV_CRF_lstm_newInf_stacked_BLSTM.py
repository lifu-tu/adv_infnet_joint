import numpy as np
import theano
from theano import tensor as T
import lasagne
import random as random
import pickle
import cPickle
import time
import sys,os
from lasagne_embedding_layer_2 import lasagne_embedding_layer_2
from random import randint

random.seed(1)
np.random.seed(1)
eps = 0.0000001

def saveParams(para, fname):
        f = file(fname, 'wb')
        cPickle.dump(para, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


def get_minibatches_idx(n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)


class GAN_CRF_model(object):

        def prepare_data(self, seqs, labels):
                lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)
                x = np.zeros((n_samples, maxlen)).astype('int32')
                x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                y = np.zeros((n_samples, maxlen)).astype('int32')
                for idx, s in enumerate(seqs):
                        x[idx,:lengths[idx]] = s
                        x_mask[idx,:lengths[idx]] = 1.
                        y[idx,:lengths[idx]] = labels[idx]

                tmp = y.flatten()
                ytmp = np.zeros((n_samples*maxlen, 25))
                ytmp[np.arange(n_samples*maxlen), tmp] = 1.0
                y_in = ytmp.reshape((n_samples, maxlen, 25)).astype('float32')
                return x, x_mask, y, y_in, maxlen
        
        def __init__(self,  We_initial,   params):
                self.textfile = open(params.outfile, 'w')
                We = theano.shared(We_initial)
                embsize = We_initial.shape[1]
                hidden = params.hidden


                lasagne.random.set_rng(np.random.RandomState(params.seed))

                l_in_word = lasagne.layers.InputLayer((None, None))
                l_mask_word = lasagne.layers.InputLayer(shape=(None, None))
                if params.emb ==1:
                        l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We)
                else:
                        l_emb_word = lasagne_embedding_layer_2(l_in_word, embsize, We)

                l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word)
                l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word, backwards = True)

                l_reshapef = lasagne.layers.ReshapeLayer(l_lstm_wordf,(-1,hidden))
                l_reshapeb = lasagne.layers.ReshapeLayer(l_lstm_wordb,(-1,hidden))
                concat2 = lasagne.layers.ConcatLayer([l_reshapef, l_reshapeb])
                l_local = lasagne.layers.DenseLayer(concat2, num_units= 25, b =None, nonlinearity=lasagne.nonlinearities.linear)
                ### the above is for the uniary term energy
    	        """
                if params.emb ==1:		
                        f = open('F.pickle')
                else:
                        f = open('F0_new.pickle')

                para = pickle.load(f)
                f.close()
                """
                f_params = lasagne.layers.get_all_params(l_local, trainable=True)
                """
                for idx, p in enumerate(f_params):
                        p.set_value(para[idx])
		"""
                Wyy0 = np.random.uniform(-0.02, 0.02, (26, 26)).astype('float32')
                Wyy = theano.shared(Wyy0)
                d_params = lasagne.layers.get_all_params(l_local, trainable=True)
                d_params.append(Wyy)
                self.d_params = d_params		
	
                l_in_word_a = lasagne.layers.InputLayer((None, None))
                l_mask_word_a = lasagne.layers.InputLayer(shape=(None, None))
                l_emb_word_a = lasagne_embedding_layer_2(l_in_word_a, embsize, l_emb_word.W)		
                #l_emb_word_a = lasagne.layers.EmbeddingLayer(l_in_word_a,  input_size=We_initial.shape[0] , output_size = embsize, W =We)
                if params.dropout:
                           l_emb_word_a = lasagne.layers.DropoutLayer(l_emb_word_a, p=0.5)

                l_lstm_wordf_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden, mask_input=l_mask_word_a)
                l_lstm_wordb_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden, mask_input=l_mask_word_a, backwards = True)
                l_lstm_feature = lasagne.layers.ConcatLayer([l_lstm_wordf_a, l_lstm_wordb_a], axis=2)
                concat2_a = lasagne.layers.ReshapeLayer(l_lstm_feature ,(-1,2*hidden))
                #l_reshapef_a = lasagne.layers.ReshapeLayer(l_lstm_wordf_a ,(-1, hidden))
                #l_reshapeb_a = lasagne.layers.ReshapeLayer(l_lstm_wordb_a ,(-1,hidden))
                #concat2_a = lasagne.layers.ConcatLayer([l_reshapef_a, l_reshapeb_a])
                #if params.dropout:
                #          concat2_a = lasagne.layers.DropoutLayer(concat2_a, p=0.5)
              
                l_local_a_inf = lasagne.layers.DenseLayer(concat2_a, num_units= 25, nonlinearity=lasagne.nonlinearities.softmax)	
		


                hidden1 = 5
                l_in_label1 = lasagne.layers.InputLayer((None, None, 25))
                l_in_label2 = lasagne.layers.InputLayer((None, None, 25))
                concat2_l = lasagne.layers.ConcatLayer([l_in_label1, l_in_label2], axis=2)
                l_lstm_wordf_l = lasagne.layers.LSTMLayer(concat2_l, hidden1, mask_input=l_mask_word_a)
                l_lstm_wordb_l = lasagne.layers.LSTMLayer(concat2_l, hidden1, mask_input=l_mask_word_a, backwards = True)
                concat2_l = lasagne.layers.ConcatLayer([l_lstm_wordf_l, l_lstm_wordb_l], axis=2)
                concat2_l = lasagne.layers.ReshapeLayer(concat2_l ,(-1,2*hidden1))
                l_local_a = lasagne.layers.DenseLayer(concat2_l, num_units= 25, nonlinearity=lasagne.nonlinearities.softmax)

		
                #a_params = lasagne.layers.get_all_params(l_local_a, trainable=True)
                #self.a_params = a_params
                """		
                if params.emb ==1:	
                        f = open('F.pickle')
                else:
                        f = open('F0_new.pickle')
                PARA = pickle.load(f)
                f.close()
               
                for idx, p in enumerate(a_params):
                        p.set_value(PARA[idx])		
	        """
                #l_local_a_inf = lasagne.layers.DenseLayer(concat2_a, num_units= 25, nonlinearity=lasagne.nonlinearities.softmax)



                y_in = T.ftensor3()
                y = T.imatrix()
                g = T.imatrix()
                gmask = T.fmatrix()
                y_mask = T.fmatrix()
                length = T.iscalar()

                predy0_inf = lasagne.layers.get_output(l_local_a_inf, {l_in_word_a:g, l_mask_word_a:gmask})
                predy_inf = predy0_inf.reshape((-1, length, 25))


   
                ###new_inf_input0 = T.concatenate([predy_inf, y_in], axis=2)
                ###new_inf_input = new_inf_input0.reshape((-1, 50))
                
                predy0 = lasagne.layers.get_output(l_local_a, {l_in_label1:predy_inf, l_in_label2:y_in, l_mask_word_a:gmask})
                predy = predy0.reshape((-1, length, 25))


                #predy = predy * gmask[:,:,None]
                #newpredy = T.concatenate([predy, y0] , axis=2)
                # n , L, 46, 46
                # predy0: n, L, 25
		
                # energy loss
                def inner_function( targets_one_step,  mask_one_step,  prev_label, tg_energy):
                        """
                        :param targets_one_step: [batch_size, t]
                        :param prev_label: [batch_size, t]
                        :param tg_energy: [batch_size]
                        :return:
                        """
                        new_ta_energy = T.dot(prev_label, Wyy[:-1,:-1])
                        new_ta_energy = tg_energy + T.sum(new_ta_energy*targets_one_step, axis =1)
                        tg_energy_t = T.switch(mask_one_step, new_ta_energy,  tg_energy)
                        return [targets_one_step, new_ta_energy]

                # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
                # but scan requires the iterable dimension to be first
                # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
                local_energy = lasagne.layers.get_output(l_local, {l_in_word: g, l_mask_word: gmask})
                local_energy = local_energy.reshape((-1, length, 25))
                local_energy = local_energy*gmask[:,:,None]
                targets_shuffled = y_in.dimshuffle(1, 0, 2)
                masks_shuffled = gmask.dimshuffle(1, 0)
                # initials should be energies_shuffles[0, :, -1, :]

                target_time0 = targets_shuffled[0]
                initial_energy0 = T.dot(target_time0, Wyy[-1,:-1])
                length_index = T.sum(gmask, axis=1)-1
                length_index = T.cast(length_index, 'int32')

                """for ground-truth energy"""
                initials = [target_time0, initial_energy0]
                [ _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[targets_shuffled[1:], masks_shuffled[1:]])
                pos_end_target = y_in[T.arange(length_index.shape[0]), length_index]
                pos_cost = target_energies[-1] + T.sum(T.sum(local_energy*y_in, axis=2)*gmask, axis=1) + T.dot( pos_end_target, Wyy[:-1,-1])    
                check = T.sum(T.sum(local_energy*y_in, axis=2)*gmask, axis=1)
                
                """for cost-augmented InfNet"""
                negtargets_shuffled = predy.dimshuffle(1, 0, 2)
                negtarget_time0 = negtargets_shuffled[0]
                neginitial_energy0 = T.dot(negtarget_time0, Wyy[-1,:-1])
                neginitials = [negtarget_time0, neginitial_energy0]
                [ _, negtarget_energies], _ = theano.scan(fn=inner_function, outputs_info=neginitials, sequences=[ negtargets_shuffled[1:], masks_shuffled[1:]])
                neg_end_target = predy[T.arange(length_index.shape[0]), length_index]
                neg_cost = negtarget_energies[-1] + T.sum(T.sum(local_energy*predy, axis=2)*gmask, axis=1) + T.dot(neg_end_target, Wyy[:-1,-1])
    	

                """for InfNet"""
                negtargets_inf_shuffled = predy_inf.dimshuffle(1, 0, 2)
                negtarget_inf_time0 = negtargets_inf_shuffled[0]
                neginitial_inf_energy0 = T.dot(negtarget_inf_time0, Wyy[-1,:-1])
                neginitials_inf = [negtarget_inf_time0, neginitial_inf_energy0]
                [ _, negtarget_inf_energies], _ = theano.scan(fn=inner_function, outputs_info=neginitials_inf, sequences=[ negtargets_inf_shuffled[1:], masks_shuffled[1:]])
                neg_inf_end_target = predy_inf[T.arange(length_index.shape[0]), length_index]
                neg_inf_cost = negtarget_inf_energies[-1] + T.sum(T.sum(local_energy*predy_inf, axis=2)*gmask, axis=1) + T.dot(neg_inf_end_target, Wyy[:-1,-1])


                y_f = y.flatten()
                predy_f =  predy.reshape((-1, 25))

                ce_hinge = lasagne.objectives.categorical_crossentropy(predy_f+eps, y_f)
                ce_hinge = ce_hinge.reshape((-1, length))
                ce_hinge = T.sum(ce_hinge* gmask, axis=1)
              
                entropy_term = - T.sum(predy_f * T.log(predy_f + eps), axis=1)
                entropy_term = entropy_term.reshape((-1, length))
                entropy_term = T.sum(entropy_term*gmask, axis=1) 		
                
                delta0 = T.sum(abs((y_in - predy)), axis=2)*gmask
                delta0 = T.sum(delta0, axis=1)


                hinge_cost_inf =  neg_inf_cost  - pos_cost

                if (params.margin_type==1):
                        hinge_cost0 = 1. + neg_cost  - pos_cost
                elif(params.margin_type==2):		
                        hinge_cost0 =  neg_cost  - pos_cost
                elif (params.margin_type==0):
                        hinge_cost0 = delta0 + neg_cost  - pos_cost
                elif(params.margin_type==3):
                        hinge_cost0 =  delta0*(1.0 + neg_cost  - pos_cost)


                #g_cost =  T.mean(T.maximum(-hinge_cost0, 0.0))


                predy_inf_f =  predy_inf.reshape((-1, 25))

           
                ce_hinge_inf = lasagne.objectives.categorical_crossentropy(predy_inf_f+eps, y_f)
                ce_hinge_inf = ce_hinge_inf.reshape((-1, length))
                ce_hinge_inf = T.sum(ce_hinge_inf* gmask, axis=1)

                

                if (params.regu_type ==0):
                        g_cost = T.mean(-hinge_cost0) + 10*T.mean(-hinge_cost_inf) + T.mean(ce_hinge_inf)
                else:
                        g_cost = T.mean(-hinge_cost0) + 10*T.mean(-hinge_cost_inf)
                
                 
 
          
 
                d_cost = T.mean(T.maximum(hinge_cost0, 0.0)) + 10*T.mean(T.maximum(hinge_cost_inf, 0.0))
                
                

                #hinge_cost = hinge_cost0 * T.gt(hinge_cost0, 0)
                #d_cost = T.sum(hinge_cost)
                #d_cost0 = d_cost				
                ###l2_term = sum(lasagne.regularization.l2(x-PARA[index]) for index, x in enumerate(a_params))

                #hinge_cost_g = hinge_cost0 * T.lt(hinge_cost0, 0)
                #d_cost0_g = T.mean(hinge_cost_g)
                
                """select different regulizer"""
                ###g_cost = -d_cost0 + params.l2* sum(lasagne.regularization.l2(x) for x in a_params) + params.l3*T.mean(ce_hinge)
                #g_cost = -d_cost0_g

                #g_cost_final = -T.mean(hinge_cost_g) + params.l2* sum(lasagne.regularization.l2(x) for x in a_params)
		
                #d_cost = d_cost

                #g_cost = -T.mean(hinge_cost_g)
                #d_cost = T.mean(hinge_cost0)

		
                a_params = lasagne.layers.get_all_params([l_local_a, l_local_a_inf], trainable=True)
                #updates_g = lasagne.updates.sgd(g_cost, a_params, params.eta)
                #updates_g = lasagne.updates.apply_momentum(updates_g, a_params, momentum=0.9)
                updates_g = lasagne.updates.adam(g_cost, a_params, 0.001)
                #updates_g_later = lasagne.updates.adam(g_cost_later, a_params, 0.0006)              

                self.a_params = a_params

                self.train_g = theano.function([g, gmask, y, y_in, length], [g_cost, d_cost, pos_cost, neg_cost, delta0, check], updates=updates_g, on_unused_input='ignore')	
                #self.train_g_later = theano.function([g, gmask, y, y_in, length], [g_cost, d_cost, pos_cost, neg_cost, delta0, check], updates=updates_g_later, on_unused_input='ignore')

                updates_d = lasagne.updates.adam(d_cost, d_params, 0.001)
                self.train_d = theano.function([g, gmask, y,  y_in, length], [d_cost, g_cost, pos_cost, neg_cost, delta0, check], updates=updates_d, on_unused_input='ignore')
               

	        """build the function for the test time inference"""
                pred = T.argmax(predy_inf, axis=2)
                pg = T.eq(pred, y)
                pg = pg*gmask
                acc_inf = 1.0* T.sum(pg)/ T.sum(gmask)

                pred = T.argmax(predy, axis=2)
                pg = T.eq(pred, y)
                pg = pg*gmask
                acc_cost = 1.0* T.sum(pg)/ T.sum(gmask)
					
                self.test_time = theano.function([g, gmask, y, length, y_in] , [acc_inf, acc_cost])


		
        def train(self, trainX, trainY, devX, devY, testX, testY, params):	

                devx0, devx0mask, devy0, devy0_in, devmaxlen = self.prepare_data(devX, devY)
                testx0, testx0mask, testy0, testy0_in, testmaxlen = self.prepare_data(testX, testY)
                devacc, _  = self.test_time(devx0, devx0mask, devy0, devmaxlen, devy0_in)
                testacc, _  = self.test_time(testx0, testx0mask, testy0, testmaxlen, testy0_in)
                self.textfile.write("initial dev acc:%f  test acc: %f  \n" %(devacc, testacc)  )		
                self.textfile.flush()
                start_time = time.time()
                bestdev = -1
                bestdev_cost = -1
                bestdev_time =0
                counter = 0
                try:
                        for eidx in xrange(500):
                                n_samples = 0
                                start_time1 = time.time()
                                kf = get_minibatches_idx(len(trainX), params.batchsize, shuffle=True)
                                uidx = 0
                                aa = 0
                                bb = 0
                                for _, train_index in kf:
                                        uidx += 1
                                        x0 = [trainX[ii] for ii in train_index]
                                        y0 = [trainY[ii] for ii in train_index]
                                        n_samples += len(train_index)
                                        x0, x0mask, y0, y0_in, maxlen = self.prepare_data(x0, y0)

                                        for ii in range(params.Lambda):
                                             #if eidx < 30:                       					
                                             g_cost, hingeloss_g, pos_cost, neg_cost, delta, pred0 = self.train_g(x0, x0mask, y0,  y0_in, maxlen)
                                             #else:
                                             #        g_cost, hingeloss_g, pos_cost, neg_cost, delta, pred0 = self.train_g_later(x0, x0mask, y0,  y0_in, maxlen)

                                        d_cost, hingeloss_d, pos_cost, neg_cost, delta, pred0 = self.train_d(x0, x0mask, y0,  y0_in, maxlen)
                                        aa += hingeloss_g
                                        bb += hingeloss_d                                
					
                                        if np.isnan(d_cost) or np.isinf(d_cost) or np.isnan(g_cost) or np.isinf(g_cost):	
                                                self.textfile.write("NaN detected \n")
                                                self.textfile.flush()
                                        #if (uidx%20==0):
                                        #         devacc, negscore, posscore, margin  = self.test_time1(devx0, devx0mask, devy0,  devy0_in, devmaxlen)
                                        #         print 'dev acc', devacc

                                end_time1 = time.time()
                                self.textfile.write("hinge loss g:%f  hinge loss d: %f    \n" %(  aa, bb)  )
                                self.textfile.flush()
                                self.textfile.write("Seen samples:%d   \n" %( n_samples)  )
                                self.textfile.flush()
                           
	
                                start_time2 = time.time()
                                devacc, devacc_cost  = self.test_time(devx0, devx0mask, devy0, devmaxlen, devy0_in)
                                end_time2 = time.time()
                                print 'dev acc', devacc
                                testacc, testacc_cost = self.test_time(testx0, testx0mask, testy0, testmaxlen, testy0_in)
                         
                                if bestdev < devacc:
                                        bestdev = devacc
                                        best_t = eidx
                                        a_para = [p.get_value() for p in self.a_params]
                                        saveParams( a_para , params.outfile + '_inf_network.pickle')


                                if bestdev_cost < devacc_cost:
                                        bestdev_cost = devacc_cost

                                self.textfile.write("epoches %d  devacc %f  testacc %f dev cost-augmented acc %f  test cost-augmented acc %f  trainig time %f test time %f \n" %( eidx + 1, devacc, testacc,devacc_cost, testacc_cost, end_time1 - start_time1, end_time2 - start_time2 ) )
                                self.textfile.flush()
                                     
        	except KeyboardInterrupt:
                        self.textfile.write( 'classifer training interrupt \n')
                        self.textfile.flush()

                end_time = time.time()
                self.textfile.write("best dev acc: %f  at time %d  cost-augment acc: %f\ \n" % (bestdev, best_t, bestdev_cost))
                self.textfile.flush()
                print 'bestdev ', bestdev		
                #os.remove(params.outfile + '_a_network.pickle')
                #self.textfile.write("best dev acc: %f  at time %d  after returning step best dev acc: %f  at time %d     \n" % (bestdev, best_t, bestdev1, best_t1))
                #self.textfile.flush()
                self.textfile.close()



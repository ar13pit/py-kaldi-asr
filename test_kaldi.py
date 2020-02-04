from kaldiasr.nnet3 import KaldiNNet3OnlineModel, KaldiNNet3OnlineDecoder

MODELDIR    = 'data/kaldi-generic-en-tdnn_fl-latest'
WAVFILE     = 'data/dw961.wav'

kaldi_model = KaldiNNet3OnlineModel(MODELDIR)
decoder     = KaldiNNet3OnlineDecoder(kaldi_model)

if decoder.decode_wav_file(WAVFILE):

    s, l = decoder.get_decoded_string()

    print()
    print("*****************************************************************")
    print(u"**", WAVFILE)
    print(u"**", s)
    print(u"** %s likelihood:" % MODELDIR, l)
    print("*****************************************************************")
    print()

else:

    print("***ERROR: decoding of %s failed." % WAVFILE)

import editdistance
def cer(r, h):
    #Remove any double or trailing
    r = ' '.join(r.split())
    h = ' '.join(h.split())

    return err(r, h)

def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    # print(float(dis) / float(len(r)))
    return float(dis) / float(len(r))

def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r,h)

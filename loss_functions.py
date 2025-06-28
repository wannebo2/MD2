def compute_loss(outputs,expectedLocs):
    dXs = outputs[0]
    dYs = outputs[1]
    dZs = outputs[2]
    expectedLocs[

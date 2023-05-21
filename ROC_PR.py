from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

ytest = np.array([])
yscore_mine = np.array([])
yscore_IP = np.array([])
yscore_offt = np.array([])

# In[11]:
# Plot ROC curves

fpr2, tpr2, thresholds = roc_curve(ytest, yscore_mine)
roc_auc2 = auc(fpr2, tpr2)

fpr1, tpr1, thresholds = roc_curve(ytest, yscore_IP)
roc_auc1 = auc(fpr1, tpr1)

fpr3, tpr3, thresholds = roc_curve(ytest, yscore_offt)
roc_auc3 = auc(fpr3, tpr3)


plt.figure()

plt.plot(fpr2, tpr2, color='red', lw=1.2, label='New model (area = %0.03f)' % roc_auc2)
plt.plot(fpr1, tpr1, color='darkblue', lw=1.2, label='CRISPR-IP (area = %0.03f)' % roc_auc1)
plt.plot(fpr3, tpr3, color='cornflowerblue', lw=1.2, label='CRISPR-offt (area = %0.03f)' % roc_auc3)


plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[12]:
# Plot PR curves

precision2, recall2, thresholds2 = precision_recall_curve(ytest, yscore_mine)
pr_auc_2 = average_precision_score(ytest, yscore_mine)

precision1, recall1, thresholds1 = precision_recall_curve(ytest, yscore_IP)
pr_auc1 = average_precision_score(ytest, yscore_IP)

precision3, recall3, thresholds3 = precision_recall_curve(ytest, yscore_offt)
pr_auc3 = average_precision_score(ytest, yscore_offt)


plt.plot(recall2, precision2, color='red', lw=1.2, label='New model (area = %0.03f)' % pr_auc_2)
plt.plot(recall1, precision1, color='darkblue', lw=1.2, label='CRISPR-IP (area = %0.03f)' % pr_auc1)
plt.plot(recall3, precision3, color='cornflowerblue', lw=1.2, label='CRISPR-offt (area = %0.03f)' % pr_auc3)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.legend(loc='upper right')
plt.show()
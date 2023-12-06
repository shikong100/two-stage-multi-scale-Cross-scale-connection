import pandas as pd

def main():
    filePath = './binary_results/pred_v.csv'
    scorePath = [
                #  'results/resnet50_ResNetBackbone_CTGNN_GAT_Fixed-Effective-MTL-version_1/resnet50_ResNetBackbone_CTGNN_GAT_Fixed-Effective-MTL-version_1_defect_valid_sigmoid.csv',
                 'results/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1_defect_valid_sigmoid.csv'
                ]
    pt = pd.read_csv(filePath, sep=',', usecols=['Filename', 'ND'])
    filename = pt[pt['ND'] == 1]['Filename']
    filename = list(filename)
    length_file = len(filename)
    temp = [0.] * length_file
    for i in scorePath:
        st = pd.read_csv(i, sep='c')
        df = pd.DataFrame({
            'Filename': filename, 
            'RB': temp, 'OB': temp, 'PF': temp, 'DE': temp, 'FS': temp, 'IS': temp, 'RO': temp, 'IN': temp, 'AF': temp, 'BE': temp, 'FO': temp, 'GR': temp, 'PH': temp, 'PB': temp, 'OS': temp, 'OP': temp, 'OK': temp
        })
        df.to_csv(i, mode='a', header=False, index=False, columns=['Filename', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK'])

if __name__ == '__main__':
    main()

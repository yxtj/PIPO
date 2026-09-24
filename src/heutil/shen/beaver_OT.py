import numpy as np
import time
import otc
# linear layers
# 0 0  0  0    y1 y2 
# 0 x1 x2 0    y3 y4                   
# 0 x3 x4 0
# 0 0  0  0

'''
0 0  0  0      B1 B2
0 A1 A2 0      B3 B4
0 A3 A4 0
0 0  0  0   


z14=x1y4=A1y4-B4*[a1]+[c1] + -B4[a1]+[c1]
z13=x1y3=A1y3-B3*[a1]+[c1] + -B3[a1]+[c1]
z24=x2y4=A2y4-B4*[a2]+[c1] + -B4[a2]+[c1]
z23..
z12..
z34..
z11..
z22..
z33..
z44..
z21..
z43..
z32..
z31..
z42..
z41..

'''
class OT:
    def __init__(self,int_width):
        self.s=otc.send()
        self.r=otc.receive()
        self.int_width=int_width
    def getPubKeyFromOther(self,key):
        self.pub_other=key
    def pubKey(self):
        return self.s.public
    def generateReplySingle(self,b_as_selection,a):
        random_vector=np.random.randint(1,2**self.int_width-1,self.int_width,'uint'+str(self.int_width*2))
        
        random=0
        reply=np.empty(self.int_width,object)
        for i in range(self.int_width):
            reply[i]=self.s.reply(b_as_selection[i],int(random_vector[i]).to_bytes(16,'big'),((a<<i)^int(random_vector[i])).to_bytes(16,'big'))
            random=random^random_vector[i]
        return (random,reply)
    def generateSelection(self,b):
        selection=np.empty(self.int_width,object)
        for i in range(self.int_width):
            selection[i]=self.r.query(self.pub_other,(b>>i)&1)
        return selection
    def decodeReplySingle(self,reply,b):
        ab_xor_r=0
        for i in range(self.int_width):
            tmp=int.from_bytes(self.r.elect(self.pub_other,(b>>i)&1,*reply[i]),'big')
            ab_xor_r=ab_xor_r^int.from_bytes(self.r.elect(self.pub_other,(b>>i)&1,*reply[i]),'big')
        return ab_xor_r
    
class BeaverEnd:

    MULTI_FACTOR=1
    TRIPLE_WIDTH=8
    TRIPLE_TYPE='uint'+str(TRIPLE_WIDTH)
    TRIPLE_TYPE_2='uint'+str(TRIPLE_WIDTH*2)

    def __init__(self,xshape,yshape,x_or_y):
              
        self.x_or_y=(x_or_y*BeaverEnd.MULTI_FACTOR).astype('int64')
        self.sign_map=np.ones(x_or_y.shape,'int8')
        for i in range(self.x_or_y.shape[0]):
            for j in range(self.x_or_y.shape[1]):
                if self.x_or_y[i][j]<0:
                    self.sign_map[i][j]=-1
                    self.x_or_y[i][j]=abs(self.x_or_y[i][j])
        
        
        self.lal_mat=np.random.randint(1,2**BeaverEnd.TRIPLE_WIDTH-1,xshape,BeaverEnd.TRIPLE_TYPE_2)
        self.lbl_mat=np.random.randint(1,2**BeaverEnd.TRIPLE_WIDTH-1,yshape,BeaverEnd.TRIPLE_TYPE_2)
        self.lcl_mat=np.zeros((xshape[0]*xshape[1],yshape[0]*yshape[1]),BeaverEnd.TRIPLE_TYPE_2)

        self.ot=OT(BeaverEnd.TRIPLE_WIDTH)
        self.random_for_lcl_genneration=np.zeros(self.lcl_mat.shape,BeaverEnd.TRIPLE_TYPE_2)
    def getPubKeyFromOther(self,key):
        self.ot.getPubKeyFromOther(key)
    def pubkey(self):
        return self.ot.pubKey()
    def generateReplyMat(self,b_as_selection_mat):
        reply_mat=np.empty(self.lcl_mat.shape,object)
        for i in range(reply_mat.shape[0]):
            for j in range(reply_mat.shape[1]):
                lal_i=int(i/self.lal_mat.shape[1])
                lal_j=i%self.lal_mat.shape[1]
                lbl_i=int(j/self.lbl_mat.shape[1])
                lbl_j=j%self.lbl_mat.shape[1]
                reply=self.ot.generateReplySingle(b_as_selection_mat[lbl_i][lbl_j],int(self.lal_mat[lal_i][lal_j]))
                self.random_for_lcl_genneration[i][j]=reply[0]
                reply_mat[i][j]=reply[1]
        return reply_mat
    def generateSelectionMat(self):
        selection_mat=np.empty(self.lbl_mat.shape,object)
        for i in range(self.lbl_mat.shape[0]):
            for j in range(self.lbl_mat.shape[1]):
                selection_mat[i][j]=self.ot.generateSelection(int(self.lbl_mat[i][j]))
        return selection_mat
    def generatelclMat(self,reply_mat):
        for i in range(self.lcl_mat.shape[0]):
            for j in range(self.lcl_mat.shape[1]):
                lal_i=int(i/self.lal_mat.shape[1])
                lal_j=i%self.lal_mat.shape[1]
                lbl_i=int(j/self.lbl_mat.shape[1])
                lbl_j=j%self.lbl_mat.shape[1]
                self.lcl_mat[i][j]=self.random_for_lcl_genneration[i][j]^(self.lal_mat[lal_i][lal_j]*self.lbl_mat[lbl_i][lbl_j])^self.ot.decodeReplySingle(reply_mat[i][j],int(self.lbl_mat[lbl_i][lbl_j]))
                #self.lcl_mat[i][j]=self.random_for_lcl_genneration[i][j]^(self.lal_mat[lal_i][lal_j]*self.lbl_mat[lbl_i][lbl_j])
  
class BeaverClient(BeaverEnd):
    
    def __init__(self,xshape,yshape,x_or_y):

        super().__init__(xshape,yshape,x_or_y)
        self.lal_mat=np.array([[1,2],[3,4]])
        self.lbl_mat=np.array([[1,1],[1,1]])
   
    def x(self):#alias
        return self.x_or_y
   
    def computeClientlzlMat(self,y_xor_server_lbl_mat: np.ndarray):
    #zij=-Bj[ai]+[cij]
        B=self.lbl_mat^y_xor_server_lbl_mat
        client_lzl=np.zeros(self.lcl_mat.shape,dtype='int64')
        for i in range(client_lzl.shape[0]):
            for j in range(client_lzl.shape[1]):
                client_lzl[i][j]=B[int(j/B.shape[1])][j%B.shape[1]]*self.lal_mat[int(i/self.lal_mat.shape[1])][i%self.lal_mat.shape[1]]^self.lcl_mat[i][j]
        return client_lzl
class BeaverServer(BeaverEnd):
    
    def __init__(self,xshape,yshape,x_or_y):

        super().__init__(xshape,yshape,x_or_y)
        self.lal_mat=np.array([[4,3],[2,1]])
        self.lbl_mat=np.array([[2,2],[2,2]])
       
    def w(self):#alias
        return self.x_or_y
    def conv(self,x_xor_client_lal_mat: np.ndarray, x_sign_map, client_lbl_mat:np.ndarray, client_lzl_mat:np.ndarray,padding: int=0, stride: int=1) -> np.ndarray:
        B_mat=self.w()^self.lbl_mat^client_lbl_mat
        A_mat=x_xor_client_lal_mat^self.lal_mat
        if padding:
            padded_A_mat=[]
            for i in range(A_mat.shape[0]+padding*2):
                padded_A_mat.append((A_mat.shape[1]+padding*2)*[None])
            i=padding
            j=padding
            for row in A_mat:
                for elemt in row:
                    padded_A_mat[i][j]=elemt
                    j=j+1
                j=padding
                i=i+1
        else:
            padded_A_mat=A_mat.tolist()

        i=0
        j=0
        result=[]
        result_index=0
        while i+self.w().shape[0]<= len(padded_A_mat):
            result.append([])
            while j+self.w().shape[1]<= len(padded_A_mat[0]):
                result[result_index].append(self.elemtMul(padded_A_mat,x_sign_map,B_mat,client_lzl_mat,i,j,padding))
                j=j+stride
            result_index=result_index+1
            j=0
            i=i+stride
        return np.array(result)

    def elemtMul(self,padded_A_mat,x_sign_map,B_mat,client_lzl_mat,i,j,padding):
        result=0
        k=j
        for wi in range(self.w().shape[0]):
            for wj in range(self.w().shape[1]):
            #zij=Aiyj-Bj*[ai]+[cij] + client_zij
                if padded_A_mat[i][k]==None:
                    pass
                else:
                    Ai=i-padding
                    Aj=k-padding
                    #result=result+(padded_A_mat[i][k]*self.w()[wi][wj]^B_mat[wi][wj]*self.lal_mat[Ai][Aj]^self.lcl_mat[Ai*self.lal_mat.shape[1]+Aj][wi*self.lbl_mat.shape[1]+wj])
                    result=result+(x_sign_map[Ai][Aj]*self.sign_map[wi][wj])*(padded_A_mat[i][k]*self.w()[wi][wj]^B_mat[wi][wj]*self.lal_mat[Ai][Aj]^self.lcl_mat[Ai*self.lal_mat.shape[1]+Aj][wi*self.lbl_mat.shape[1]+wj]^client_lzl_mat[Ai*self.lal_mat.shape[1]+Aj][wi*self.lbl_mat.shape[1]+wj])
                    result=float(result)/(BeaverServer.MULTI_FACTOR*BeaverServer.MULTI_FACTOR)
                k=k+1
            k=j
            i=i+1
        return result
    
if __name__ == "__main__":
    
    x=np.array([[1.0,2.0],[3.0,4.0]])
    w=np.array([[1,-1],[1,1]])

    start=time.time()
    server=BeaverServer(x.shape,w.shape,w)
    client=BeaverClient(x.shape,w.shape,x)
    
    server.getPubKeyFromOther(client.pubkey())
    client.getPubKeyFromOther(server.pubkey())
    server_reply=server.generateReplyMat(client.generateSelectionMat())
    client_reply=client.generateReplyMat(server.generateSelectionMat())
    client.generatelclMat(server_reply)
    server.generatelclMat(client_reply)

    '''
    print(client.lal_mat^server.lal_mat)
    print(client.lbl_mat^server.lbl_mat)
    
    print(client.lcl_mat)
    print(server.lcl_mat)
    print(client.lcl_mat^server.lcl_mat)
    '''
    client_lzl_mat=client.computeClientlzlMat(server.w()^server.lbl_mat)
    
    res=server.conv(client.x()^client.lal_mat,client.sign_map,client.lbl_mat,client_lzl_mat,1)
    end=time.time()
    print(res)
    print(end-start)
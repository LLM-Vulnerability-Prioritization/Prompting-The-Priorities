# Prompting Techniques: Different prompting techniques to determine if prompting technique impacts LLM performance
prompt_techniques = {
                    "zero_shot": 
                        """
                        <Instruction>
                            Analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                        """,
                     
                     "one_shot": 
                         """
                        <Example>
                           A vulnerability with the following details:
                            <AttackerKbDescription>Critical remote code execution vulnerability in ACME Web Server login page</AttackerKbDescription>
                            <AttackerKbTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released 2022-01-20: Public disclosure </AttackerKbTimeline>
                            <AttackerKbVulnerableVersions>ACME Web Server 2.0.0 - 2.5.0</AttackerKbVulnerableVersions>
                            <AttackerKbVendorProductNames>ACME Software - ACME Web Server</AttackerKbVendorProductNames>
                            <AttackerKbTags>Remote Code Execution, Web Application, Critical</AttackerKbTags>
                            <AttackerKbMitreTactics>Execution, Lateral Movement, Credential Access</AttackerKbMitreTactics>                         
                            <BugtraqAdvisory>The vulnerability allows unauthenticated attackers to execute arbitrary commands on the server with root privileges. Successful exploitation can lead to complete system compromise.</BugtraqAdvisory>                          
                            <KevVendor>ACME Software</KevVendor> 
                            <KevProduct>ACME Web Server</KevProduct>
                            <KevVulnerabilityName>ACME Web Server Remote Code Execution</KevVulnerabilityName> 
                            <KevShortDescription>A critical vulnerability in ACME Web Server allows remote attackers to execute arbitrary code on the server.</KevShortDescription> 
                            <KevRequiredAction>Immediately patch affected systems or disconnect them from the network until a patch can be applied.</KevRequiredAction> 
                            <KevKnownRansomwareCampaignUse>The RansomX group is actively exploiting this vulnerability to deploy ransomware.</KevKnownRansomwareCampaignUse>                           
                            <CveDescription>ACME Web Server before 2.5.1 allows remote attackers to execute arbitrary code via a crafted request to the login page.</CveDescription>                           
                            <ExploitDescription>The exploit takes advantage of an OS command injection flaw in the login page to execute malicious commands.</ExploitDescription>
                            <ExploitContent>
                                $payload = "admin;phpinfo();";
                                $url = "http://example.com/index.php?user=" . urlencode($payload);
                                $response = file_get_contents($url);
                                echo $response;
                            </ExploitContent>                           
                            <FulldisAdvisoryEmail>A patch has been released to address this critical vulnerability. All customers are urged to upgrade immediately.</FulldisAdvisoryEmail>                          
                            <LinuxVulDescriptions>The vulnerability affects all Linux versions of ACME Web Server from 2.0.0 to 2.5.0.</LinuxVulDescriptions> <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle> <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>                           
                            <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle>
                            <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>  
                            <OssAdvisoryEmail>A critical vulnerability has been discovered in ACME Web Server. All users are urged to upgrade immediately to version 2.5.1 or later.</OssAdvisoryEmail>
                            <PatchCode>echo ($user = filter_var($_GET["user"], FILTER_SANITIZE_STRING)) ? htmlspecialchars($user, ENT_QUOTES, 'UTF-8') : exit('Invalid user input');</PatchCode>
                            <ZdiTitle>ACME Web Server login page Remote Code Execution Vulnerability</ZdiTitle> 
                            <ZdiVendors>ACME Software</ZdiVendors> <ZdiProducts>ACME Web Server</ZdiProducts> 
                            <ZdiProducts>ACME Web Server</ZdiProducts> 
                            <ZdiDescription>This vulnerability allows remote attackers to execute arbitrary code on affected installations of ACME Web Server. Authentication is not required to exploit this vulnerability.</ZdiDescription> 
                            <ZdiDetails>The specific flaw exists within the processing of the login page parameters. The issue results from the lack of proper validation of user-supplied data, which can allow for OS command injection. An attacker can leverage this vulnerability to execute code in the context of the web server.</ZdiDetails> 
                            <ZdiTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released 2022-01-20: Public disclosure </ZdiTimeline>
                           returns: {2}
                        </Example>
                        <Instruction>
                            Given this, analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                         """,

                     "few_shot": 
                         """
                         <Example>
                            A vulnerability with the following details:
                            <AttackerKbDescription>Critical remote code execution vulnerability in ACME Web Server login page</AttackerKbDescription>
                            <AttackerKbTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released  2022-01-20: Public disclosure</AttackerKbTimeline>
                            <AttackerKbVulnerableVersions>ACME Web Server 2.0.0 - 2.5.0</AttackerKbVulnerableVersions>
                            <AttackerKbVendorProductNames>ACME Software - ACME Web Server</AttackerKbVendorProductNames>
                            <AttackerKbTags>Remote Code Execution, Web Application, Critical</AttackerKbTags>
                            <AttackerKbMitreTactics>Execution, Lateral Movement, Credential Access</AttackerKbMitreTactics>                         
                            <BugtraqAdvisory>The vulnerability allows unauthenticated attackers to execute arbitrary commands on the server with root privileges. Successful exploitation can lead to complete system compromise.</BugtraqAdvisory>                          
                            <KevVendor>ACME Software</KevVendor> 
                            <KevProduct>ACME Web Server</KevProduct>
                            <KevVulnerabilityName>ACME Web Server Remote Code Execution</KevVulnerabilityName> 
                            <KevShortDescription>A critical vulnerability in ACME Web Server allows remote attackers to execute arbitrary code on the server.</KevShortDescription> 
                            <KevRequiredAction>Immediately patch affected systems or disconnect them from the network until a patch can be applied.</KevRequiredAction> 
                            <KevKnownRansomwareCampaignUse>The RansomX group is actively exploiting this vulnerability to deploy ransomware.</KevKnownRansomwareCampaignUse>                           
                            <CveDescription>ACME Web Server before 2.5.1 allows remote attackers to execute arbitrary code via a crafted request to the login page.</CveDescription>                           
                            <ExploitDescription>The exploit takes advantage of an OS command injection flaw in the login page to execute malicious commands.</ExploitDescription>
                            <ExploitContent>
                                $payload = "admin;phpinfo();";
                                $url = "http://example.com/index.php?user=" . urlencode($payload);
                                $response = file_get_contents($url);
                                echo $response;
                            </ExploitContent>                           
                            <FulldisAdvisoryEmail>A patch has been released to address this critical vulnerability. All customers are urged to upgrade immediately.</FulldisAdvisoryEmail>                          
                            <LinuxVulDescriptions>The vulnerability affects all Linux versions of ACME Web Server from 2.0.0 to 2.5.0.</LinuxVulDescriptions> <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle> <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>                           
                            <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle>
                            <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>
                            <OssAdvisoryEmail>A critical vulnerability has been discovered in ACME Web Server. All users are urged to upgrade immediately to version 2.5.1 or later.</OssAdvisoryEmail>
                            <PatchCode>echo ($user = filter_var($_GET["user"], FILTER_SANITIZE_STRING)) ? htmlspecialchars($user, ENT_QUOTES, 'UTF-8') : exit('Invalid user input');</PatchCode>
                            <ZdiTitle>ACME Web Server login page Remote Code Execution Vulnerability</ZdiTitle> 
                            <ZdiVendors>ACME Software</ZdiVendors> <ZdiProducts>ACME Web Server</ZdiProducts> 
                            <ZdiDescription>This vulnerability allows remote attackers to execute arbitrary code on affected installations of ACME Web Server. Authentication is not required to exploit this vulnerability.</ZdiDescription> 
                            <ZdiDetails>The specific flaw exists within the processing of the login page parameters. The issue results from the lack of proper validation of user-supplied data, which can allow for OS command injection. An attacker can leverage this vulnerability to execute code in the context of the web server.</ZdiDetails> 
                            <ZdiTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released 2022-01-20: Public disclosure </ZdiTimeline>
                            returns: {2}
                        </Example>
                        <Example>
                           A vulnerability with the following details:
                            <AttackerKbDescription>Information disclosure vulnerability in Example File Sharing App allows authenticated users to view unauthorized files</AttackerKbDescription>
                            <AttackerKbTimeline> 2022-06-01: Vulnerability discovered 2022-06-10: Vendor notified 2022-06-15: Patch released (version 1.2.3)</AttackerKbTimeline>
                            <AttackerKbVulnerableVersions>Example File Sharing App 1.0.0 - 1.2.2</AttackerKbVulnerableVersions>
                            <AttackerKbVendorProductNames>Example Company - Example File Sharing App</AttackerKbVendorProductNames>
                            <AttackerKbTags>Information Disclosure, Access Control, Authenticated</AttackerKbTags>
                            <AttackerKbMitreTactics>Discovery</AttackerKbMitreTactics>
                            <BugtraqAdvisory>An information disclosure vulnerability exists that allows low privileged users to view sensitive files. Exploitation requires a valid user account.</BugtraqAdvisory>
                            <KevVendor>Example Company</KevVendor>
                            <KevProduct>Example File Sharing App</KevProduct>
                            <KevVulnerabilityName>Example File Sharing App Improper Access Control</KevVulnerabilityName>
                            <KevShortDescription>The file sharing application allows users to view files they should not have access to.</KevShortDescription>
                            <KevRequiredAction>Upgrade to the latest version which includes a fix for this vulnerability.</KevRequiredAction>
                            <KevKnownRansomwareCampaignUse>None</KevKnownRansomwareCampaignUse>
                            <CveDescription>Example File Sharing App before 1.2.3 does not properly check user permissions, allowing authenticated users to view any file.</CveDescription>
                            <ExploitDescription>An attacker can manipulate the fileID parameter to view files belonging to other users.</ExploitDescription> 
                            <ExploitContent>String response = new BufferedReader(new InputStreamReader(((HttpURLConnection) new URL("http://example.com/download?fileId=" + URLEncoder.encode("../../etc/passwd", "UTF-8")).openConnection()).getInputStream())).lines().collect(Collectors.joining("\n"));</ExploitContent>
                            <FulldisAdvisoryEmail>A low severity information disclosure vulnerability has been patched in the latest release. Users should upgrade when possible.</FulldisAdvisoryEmail>
                            <LinuxVulDescriptions>The improper access control vulnerability affects the Linux version of the file sharing software.</LinuxVulDescriptions> 
                            <LinuxVulTitle>Example File Sharing App 1.0.0-1.2.2 Improper Access Control on Linux</LinuxVulTitle> 
                            <LinuxVulEmailAdvisory>Linux users of the file sharing app should upgrade to version 1.2.3 to fix an information disclosure vulnerability.</LinuxVulEmailAdvisory>
                            <OssAdvisoryEmail>A low severity information disclosure vulnerability has been discovered in Example File Sharing App versions 1.0.0 - 1.2.2. Users are advised to upgrade to version 1.2.3 which contains a fix for this issue.</OssAdvisoryEmail>
                            <PatchCode>File file = (user.hasAccessTo((fileId = Integer.parseInt(request.getParameter("fileId")))) ? getFileById(fileId) : throw new AccessDeniedException("Access denied"));</PatchCode>
                            <ZdiTitle>Example File Sharing App fileID Parameter Improper Access Control</ZdiTitle>
                            <ZdiVendors>Example Company</ZdiVendors>
                            <ZdiProducts>Example File Sharing App</ZdiProducts>
                            <ZdiDescription>This vulnerability allows authenticated users to view files they should not have access to.</ZdiDescription> 
                            <ZdiDetails>The application does not properly check that the requesting user has permission to view the file specified by the fileID parameter. By modifying the fileID to a file owned by a different user, an attacker can retrieve files they should not have access to.</ZdiDetails> 
                            <ZdiTimeline> 2022-06-01: Vulnerability reported by researcher 2022-06-10: Fix developed by vendor 2022-06-15: Version 1.2.3 released to address vulnerability </ZdiTimeline>
                           returns: {3}
                        </Example>
                        <Instruction>
                            Given this, analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                         """,                        

                     "recursive_criticism_improvement": 
                         """
                        <Instruction>
                            Analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                            Critique your response. Then propose an optimised response based upon your
                            critique.
                        </Instruction>
                         """,
                     
                     "self_refine": 
                         """
                        <Instruction> 
                            Analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                            Then generate feedback on your response, and then use this feedback to generate an
                            optimised response. Repeat the feedback and optimisation process until you are
                            satisfied with your response, but do not exceed three iterations of this
                            process.
                        </Instruction>
                         """,
                     
                     "self_planning": 
                         """
                        <Instruction>
                            Describe how you would {0} based on vulnerability data. Based on this description, then analyse
                            the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                            """,
                     
                     "chain_of_thought": 
                         """
                        <Example>
                               A vulnerability with the following details:
                            <AttackerKbDescription>Critical remote code execution vulnerability in ACME Web Server login page</AttackerKbDescription>
                            <AttackerKbTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released  2022-01-20: Public disclosure</AttackerKbTimeline>
                            <AttackerKbVulnerableVersions>ACME Web Server 2.0.0 - 2.5.0</AttackerKbVulnerableVersions>
                            <AttackerKbVendorProductNames>ACME Software - ACME Web Server</AttackerKbVendorProductNames>
                            <AttackerKbTags>Remote Code Execution, Web Application, Critical</AttackerKbTags>
                            <AttackerKbMitreTactics>Execution, Lateral Movement, Credential Access</AttackerKbMitreTactics>                         
                            <BugtraqAdvisory>The vulnerability allows unauthenticated attackers to execute arbitrary commands on the server with root privileges. Successful exploitation can lead to complete system compromise.</BugtraqAdvisory>                          
                            <KevVendor>ACME Software</KevVendor> 
                            <KevProduct>ACME Web Server</KevProduct>
                            <KevVulnerabilityName>ACME Web Server Remote Code Execution</KevVulnerabilityName> 
                            <KevShortDescription>A critical vulnerability in ACME Web Server allows remote attackers to execute arbitrary code on the server.</KevShortDescription> 
                            <KevRequiredAction>Immediately patch affected systems or disconnect them from the network until a patch can be applied.</KevRequiredAction> 
                            <KevKnownRansomwareCampaignUse>The RansomX group is actively exploiting this vulnerability to deploy ransomware.</KevKnownRansomwareCampaignUse>                           
                            <CveDescription>ACME Web Server before 2.5.1 allows remote attackers to execute arbitrary code via a crafted request to the login page.</CveDescription>                           
                            <ExploitDescription>The exploit takes advantage of an OS command injection flaw in the login page to execute malicious commands.</ExploitDescription>
                            <ExploitContent>
                                $payload = "admin;phpinfo();";
                                $url = "http://example.com/index.php?user=" . urlencode($payload);
                                $response = file_get_contents($url);
                                echo $response;
                            </ExploitContent>                           
                            <FulldisAdvisoryEmail>A patch has been released to address this critical vulnerability. All customers are urged to upgrade immediately.</FulldisAdvisoryEmail>                          
                            <LinuxVulDescriptions>The vulnerability affects all Linux versions of ACME Web Server from 2.0.0 to 2.5.0.</LinuxVulDescriptions> <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle> <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>                           
                            <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle>
                            <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>
                            <OssAdvisoryEmail>A critical vulnerability has been discovered in ACME Web Server versions 2.0.0 - 2.5.0. All users are urged to upgrade immediately to version 2.5.1 or later.</OssAdvisoryEmail>
                            <PatchCode>echo ($user = filter_var($_GET["user"], FILTER_SANITIZE_STRING)) ? htmlspecialchars($user, ENT_QUOTES, 'UTF-8') : exit('Invalid user input');</PatchCode>
                            <ZdiTitle>ACME Web Server login page Remote Code Execution Vulnerability</ZdiTitle> 
                            <ZdiVendors>ACME Software</ZdiVendors> <ZdiProducts>ACME Web Server</ZdiProducts> 
                            <ZdiProducts>ACME Web Server</ZdiProducts>
                            <ZdiDescription>This vulnerability allows remote attackers to execute arbitrary code on affected installations of ACME Web Server. Authentication is not required to exploit this vulnerability.</ZdiDescription> 
                            <ZdiDetails>The specific flaw exists within the processing of the login page parameters. The issue results from the lack of proper validation of user-supplied data, which can allow for OS command injection. An attacker can leverage this vulnerability to execute code in the context of the web server.</ZdiDetails> 
                            <ZdiTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released 2022-01-20: Public disclosure </ZdiTimeline>
                               returns: {2}
                               <Explanation>
                                   The vulnerability allows unauthenticated remote code execution, which is one of the most critical types of security flaws. This type of vulnerability can lead to complete system compromise, allowing attackers to gain full control over the affected systems. The public availability of exploit code and active exploitation by threat actors, including ransomware groups, further elevates the risk, making it a critical issue that requires immediate attention and remediation. The potential impact is severe, with the possibility of significant data loss, financial damage, and operational disruption.
                               </Explanation>
                        </Example>
                        <Instruction> 
                            Given this, analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                         """,
                     
                     "zero_shot_chain_of_thought": 
                         """
                         <Instruction>
                             Analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                            Think about this step by step.
                        </Instruction>
                         """,
                     
                     "self_consistency": 
                         """
                         <Instruction>
                             Perform multiple analyses of the details of this vulnerability and {0}:                        
                            <VulnerabilityData>{1}</VulnerabilityData>
                            Determine the most common response of your multiple analyses and return this response.
                        </Instruction>
                         """,
                     
                     "few_shot_with_explanation": 
                         """
                        <Example>
                               A vulnerability with the following details:
                            <AttackerKbDescription>Critical remote code execution vulnerability in ACME Web Server login page</AttackerKbDescription>
                            <AttackerKbTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released 2022-01-20: Public disclosure</AttackerKbTimeline>
                            <AttackerKbVulnerableVersions>ACME Web Server 2.0.0 - 2.5.0</AttackerKbVulnerableVersions>
                            <AttackerKbVendorProductNames>ACME Software - ACME Web Server</AttackerKbVendorProductNames>
                            <AttackerKbTags>Remote Code Execution, Web Application, Critical</AttackerKbTags>
                            <AttackerKbMitreTactics>Execution, Lateral Movement, Credential Access</AttackerKbMitreTactics>                         
                            <BugtraqAdvisory>The vulnerability allows unauthenticated attackers to execute arbitrary commands on the server with root privileges. Successful exploitation can lead to complete system compromise.</BugtraqAdvisory>                          
                            <KevVendor>ACME Software</KevVendor> 
                            <KevProduct>ACME Web Server</KevProduct>
                            <KevVulnerabilityName>ACME Web Server Remote Code Execution</KevVulnerabilityName> 
                            <KevShortDescription>A critical vulnerability in ACME Web Server allows remote attackers to execute arbitrary code on the server.</KevShortDescription> 
                            <KevRequiredAction>Immediately patch affected systems or disconnect them from the network until a patch can be applied.</KevRequiredAction> 
                            <KevKnownRansomwareCampaignUse>The RansomX group is actively exploiting this vulnerability to deploy ransomware.</KevKnownRansomwareCampaignUse>                           
                            <CveDescription>ACME Web Server before 2.5.1 allows remote attackers to execute arbitrary code via a crafted request to the login page.</CveDescription>                           
                            <ExploitDescription>The exploit takes advantage of an OS command injection flaw in the login page to execute malicious commands.</ExploitDescription>
                            <ExploitContent>
                                $payload = "admin;phpinfo();";
                                $url = "http://example.com/index.php?user=" . urlencode($payload);
                                $response = file_get_contents($url);
                                echo $response;
                            </ExploitContent>                           
                            <FulldisAdvisoryEmail>A patch has been released to address this critical vulnerability. All customers are urged to upgrade immediately.</FulldisAdvisoryEmail>                          
                            <LinuxVulDescriptions>The vulnerability affects all Linux versions of ACME Web Server from 2.0.0 to 2.5.0.</LinuxVulDescriptions> <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle> <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>                           
                            <LinuxVulTitle>ACME Web Server 2.0.0-2.5.0 Remote Code Execution</LinuxVulTitle>
                            <LinuxVulEmailAdvisory>Linux administrators should prioritize patching this vulnerability on affected servers.</LinuxVulEmailAdvisory>
                            <OssAdvisoryEmail>A critical vulnerability has been discovered in ACME Web Server versions 2.0.0 - 2.5.0. All users are urged to upgrade immediately to version 2.5.1 or later.</OssAdvisoryEmail>
                            <PatchCode>echo ($user = filter_var($_GET["user"], FILTER_SANITIZE_STRING)) ? htmlspecialchars($user, ENT_QUOTES, 'UTF-8') : exit('Invalid user input');</PatchCode>
                            <ZdiTitle>ACME Web Server login page Remote Code Execution Vulnerability</ZdiTitle> 
                            <ZdiVendors>ACME Software</ZdiVendors> <ZdiProducts>ACME Web Server</ZdiProducts> 
                            <ZdiProducts>ACME Web Server</ZdiProducts>
                            <ZdiDescription>This vulnerability allows remote attackers to execute arbitrary code on affected installations of ACME Web Server. Authentication is not required to exploit this vulnerability.</ZdiDescription> 
                            <ZdiDetails>The specific flaw exists within the processing of the login page parameters. The issue results from the lack of proper validation of user-supplied data, which can allow for OS command injection. An attacker can leverage this vulnerability to execute code in the context of the web server.</ZdiDetails> 
                            <ZdiTimeline> 2022-01-01: Vulnerability discovered 2022-01-02: Vendor notified 2022-01-15: Patch released 2022-01-20: Public disclosure </ZdiTimeline>
                               returns: {2}
                               <Explanation>
                                   The vulnerability allows unauthenticated remote code execution, which is one of the most critical types of security flaws. This type of vulnerability can lead to complete system compromise, allowing attackers to gain full control over the affected systems. The public availability of exploit code and active exploitation by threat actors, including ransomware groups, further elevates the risk, making it a critical issue that requires immediate attention and remediation. The potential impact is severe, with the possibility of significant data loss, financial damage, and operational disruption.
                               </Explanation>
                        </Example>
                        <Example>
                               A vulnerability with the following details:
                            <AttackerKbDescription>Information disclosure vulnerability in Example File Sharing App allows authenticated users to view unauthorized files</AttackerKbDescription>
                            <AttackerKbTimeline> 2022-06-01: Vulnerability discovered 2022-06-10: Vendor notified 2022-06-15: Patch released (version 1.2.3) </AttackerKbTimeline>
                            <AttackerKbVulnerableVersions>Example File Sharing App 1.0.0 - 1.2.2</AttackerKbVulnerableVersions>
                            <AttackerKbVendorProductNames>Example Company - Example File Sharing App</AttackerKbVendorProductNames>
                            <AttackerKbTags>Information Disclosure, Access Control, Authenticated</AttackerKbTags>
                            <AttackerKbMitreTactics>Discovery</AttackerKbMitreTactics>
                            <BugtraqAdvisory>An information disclosure vulnerability exists that allows low privileged users to view sensitive files. Exploitation requires a valid user account.</BugtraqAdvisory>
                            <KevVendor>Example Company</KevVendor> <KevProduct>Example File Sharing App</KevProduct> <KevVulnerabilityName>Example File Sharing App Improper Access Control</KevVulnerabilityName>
                            <KevShortDescription>The file sharing application allows users to view files they should not have access to.</KevShortDescription> 
                            <KevRequiredAction>Upgrade to the latest version which includes a fix for this vulnerability.</KevRequiredAction> 
                            <KevKnownRansomwareCampaignUse>None</KevKnownRansomwareCampaignUse>
                            <CveDescription>Example File Sharing App before 1.2.3 does not properly check user permissions, allowing authenticated users to view any file.</CveDescription>
                            <ExploitDescription>An attacker can manipulate the fileID parameter to view files belonging to other users.</ExploitDescription> 
                            <ExploitContent>String response = new BufferedReader(new InputStreamReader(((HttpURLConnection) new URL("http://example.com/download?fileId=" + URLEncoder.encode("../../etc/passwd", "UTF-8")).openConnection()).getInputStream())).lines().collect(Collectors.joining("\n"));</ExploitContent>
                            <FulldisAdvisoryEmail>A low severity information disclosure vulnerability has been patched in the latest release. Users should upgrade when possible.</FulldisAdvisoryEmail>
                            <LinuxVulDescriptions>The improper access control vulnerability affects the Linux version of the file sharing software.</LinuxVulDescriptions> 
                            <LinuxVulTitle>Example File Sharing App 1.0.0-1.2.2 Improper Access Control on Linux</LinuxVulTitle> 
                            <LinuxVulEmailAdvisory>Linux users of the file sharing app should upgrade to version 1.2.3 to fix an information disclosure vulnerability.</LinuxVulEmailAdvisory>
                            <OssAdvisoryEmail>A low severity information disclosure vulnerability has been discovered in Example File Sharing App versions 1.0.0 - 1.2.2. Users are advised to upgrade to version 1.2.3 which contains a fix for this issue.</OssAdvisoryEmail>
                            <PatchCode>File file = (user.hasAccessTo((fileId = Integer.parseInt(request.getParameter("fileId")))) ? getFileById(fileId) : throw new AccessDeniedException("Access denied"));</PatchCode>
                            <ZdiTitle>Example File Sharing App fileID Parameter Improper Access Control</ZdiTitle>
                            <ZdiVendors>Example Company</ZdiVendors>
                            <ZdiProducts>Example File Sharing App</ZdiProducts>
                            <ZdiDescription>This vulnerability allows authenticated users to view files they should not have access to.</ZdiDescription> 
                            <ZdiDetails>The application does not properly check that the requesting user has permission to view the file specified by the fileID parameter. By modifying the fileID to a file owned by a different user, an attacker can retrieve files they should not have access to.</ZdiDetails> 
                            <ZdiTimeline> 2022-06-01: Vulnerability reported by researcher 2022-06-10: Fix developed by vendor 2022-06-15: Version 1.2.3 released to address vulnerability </ZdiTimeline>
                               returns: {3}
                               <Explanation>
                                   The vulnerability involves an information disclosure flaw that requires an authenticated user account to exploit. This limits the potential attackers to those who already have some level of access, reducing the likelihood of exploitation compared to vulnerabilities that can be exploited by any anonymous user. The impact is also more limited, as it involves unauthorized access to files rather than full system compromise. There are no reports of active exploitation, and the vendor rates it as low severity, indicating that while it should be addressed, it does not pose an immediate threat. This makes it a lower priority compared to more critical vulnerabilities.
                               </Explanation>
                        </Example>
                        <Instruction>
                            Given this, analyse the details of this vulnerability and {0}:
                            <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                         """,
                     
                     "persona_pattern": 
                         """
                        <Context>Assume you are a Security Operations Centre (SOC) analyst.</Context>
                        <Instruction>Given this, analyse the details of this vulnerability and {0}:
                        <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                         """,
                     
                     "memetic_proxy": 
                         """ 
                         <Context>Assume you are the former director of CISA (Cybersecurity and Infrastructure Security Agency), known for extensive experience in protecting critical infrastructure and managing national cybersecurity risks.</Context>
                        <Instruction>Given this, analyse the details of this vulnerability and {0}:
                        <VulnerabilityData>{1}</VulnerabilityData>
                        </Instruction>
                         """,
                     }    